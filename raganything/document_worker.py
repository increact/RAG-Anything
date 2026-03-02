"""
Document processing worker for handling queued tasks
"""
import os
import asyncio
import tempfile
import logging
import httpx
from typing import Dict, Any
from pathlib import Path
from urllib.parse import urlparse, unquote

logger = logging.getLogger(__name__)

# Global RAG instance (will be set by worker initialization)
rag_instance = None


def set_rag_instance(instance):
    """Set the global RAG instance"""
    global rag_instance
    rag_instance = instance


async def process_document_task_async(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process document task asynchronously
    
    Args:
        task_data: Task data containing document_id, s3_url, processing_options, etc.
    
    Returns:
        Dict with processing results
    """
    task_id = task_data.get("task_id", "unknown")
    document_id = task_data.get("document_id", "")
    project_id = task_data.get("project_id", "")
    webhook_url = task_data.get("webhook_url", "")

    temp_file = None

    try:
        s3_url = task_data["s3_url"]
        processing_options = task_data["processing_options"]

        logger.info(f"Processing task {task_id} for document {document_id}")
        
        if rag_instance is None:
            raise Exception("RAG instance not initialized")
        
        # Extract file extension from URL path before downloading
        parsed_url = urlparse(s3_url)
        url_path = unquote(parsed_url.path)  # Decode URL encoding
        file_ext = Path(url_path).suffix or ".pdf"

        # Stream download directly to a temp file to avoid loading the
        # entire file into memory (important for large documents).
        logger.info(f"Downloading file from: {s3_url}")
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
            temp_file = tmp.name
            async with httpx.AsyncClient(timeout=300.0) as client:
                async with client.stream("GET", s3_url) as response:
                    response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=65536):
                        tmp.write(chunk)
        
        logger.info(f"File saved to temporary location: {temp_file}")
        
        # Process document
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)
        
        parser = processing_options.get("parser", "auto")
        parse_method = processing_options.get("parse_method", "auto")
        language = processing_options.get("language", "zh")
        device = processing_options.get("device", "cpu")
        
        # formula/table detection each load a heavy ML model (~700MB-1GB each).
        # Default to env-var controlled values so memory usage can be tuned at
        # deployment time without changing caller code.
        # Callers can always override per-request via processing_options.
        formula_default = os.getenv("MINERU_FORMULA_DEFAULT", "false").lower() == "true"
        table_default = os.getenv("MINERU_TABLE_DEFAULT", "false").lower() == "true"

        parser_kwargs = {
            "lang": language,
            "device": device,
            "formula": processing_options.get("formula", formula_default),
            "table": processing_options.get("table", table_default),
            "backend": processing_options.get("backend", "pipeline"),
        }
        
        # If session_id is provided, add to parser_kwargs
        if "session_id" in processing_options:
            parser_kwargs["session_id"] = processing_options["session_id"]
        
        # Auto-select parser
        if parser == "auto":
            file_ext = Path(temp_file).suffix.lower()
            if file_ext in [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]:
                selected_parser = "docling"
            else:
                selected_parser = "mineru"
        else:
            selected_parser = parser
        
        # Update config
        rag_instance.config.parser = selected_parser
        
        logger.info(f"Parsing document with parser: {selected_parser}, method: {parse_method}")
        
        # Parse document
        content_list, doc_id = await rag_instance.parse_document(
            temp_file,
            output_dir,
            parse_method,
            display_stats=True,
            **parser_kwargs,
        )
        
        logger.info(f"Document parsed successfully, doc_id: {doc_id}")
        
        # Read markdown output (written by the parser to the output dir)
        file_stem = Path(temp_file).stem
        md_file = Path(output_dir) / f"{file_stem}.md"
        markdown = ""
        if md_file.exists():
            with open(md_file, "r", encoding="utf-8") as f:
                markdown = f.read()
        
        # Build basic metadata counts
        metadata: dict = {
            "parser": selected_parser,
            "parse_method": parse_method,
            "doc_id": doc_id,
            "tables": 0,
            "formulas": 0,
            "images": 0,
        }
        for item in content_list:
            if isinstance(item, dict):
                t = item.get("type", "")
                if t == "table":
                    metadata["tables"] += 1
                elif t == "equation":
                    metadata["formulas"] += 1
                elif t == "image":
                    metadata["images"] += 1

        # Persist result to disk so GET /api/v1/result/{doc_id} can serve it
        from raganything.result_store import save_result
        save_result(
            output_dir=output_dir,
            doc_id=doc_id,
            document_id=document_id,
            project_id=project_id,
            markdown=markdown,
            content_list=content_list,
            metadata=metadata,
        )

        # Fix webhook URL for Docker networking
        fixed_webhook_url = webhook_url.replace("localhost", "host.docker.internal")
        if fixed_webhook_url != webhook_url:
            logger.info(f"Fixed webhook URL for Docker: {webhook_url} -> {fixed_webhook_url}")
        
        # Send a lightweight notification webhook — no content payload.
        # The receiver fetches the full result via GET /api/v1/result/{doc_id}.
        from raganything.webhook_service import WebhookService
        
        webhook_success = await WebhookService.send_webhook_with_retry(
            webhook_url=fixed_webhook_url,
            doc_id=doc_id,
            document_id=document_id,
            project_id=project_id,
            status="completed",
            max_retries=3,
        )
        
        if not webhook_success:
            logger.warning(
                f"⚠️ Webhook failed for task {task_id} but document processing succeeded. "
                f"Document ID: {doc_id}. Client can fetch results via GET /api/v1/result/{doc_id}."
            )
        
        return {
            "success": True,
            "doc_id": doc_id,
            "webhook_sent": webhook_success,
        }
        
    except Exception as e:
        logger.error(f"Error processing task {task_id}: {str(e)}")
        logger.exception(e)
        # Do NOT send the failure webhook here.
        # With RQ Retry(max=3) the job can be retried up to 3 times; sending
        # the webhook on every failed attempt would spam the client.
        # handle_job_failure() is invoked by RQ once ALL retries are exhausted
        # and is responsible for the single definitive failure notification.
        raise
    
    finally:
        # Clean up temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file {temp_file}: {e}")


def handle_job_failure(job, connection, type, value, traceback_obj):
    """
    RQ on_failure callback — called once when ALL retries are exhausted.

    This is the single place that sends the failure webhook to the client,
    regardless of whether the failure was a timeout or any other exception.
    Keeping webhook delivery here avoids sending duplicate notifications when
    the job is retried (Retry(max=3) would otherwise trigger the webhook on
    every failed attempt).
    """
    try:
        task_data = job.args[0] if job.args else {}
        webhook_url = task_data.get("webhook_url", "")
        document_id = task_data.get("document_id", "")
        project_id = task_data.get("project_id", "")

        if not webhook_url:
            logger.warning(f"Job {job.id} failed but no webhook_url to notify")
            return

        error_msg = str(value) if value else "Unknown error"
        logger.error(
            f"Job {job.id} (document: {document_id}) permanently failed: {error_msg}. "
            f"Sending failure webhook."
        )

        fixed_webhook_url = webhook_url.replace("localhost", "host.docker.internal")
        if fixed_webhook_url != webhook_url:
            logger.info(f"Fixed webhook URL for Docker: {webhook_url} -> {fixed_webhook_url}")

        send_loop = asyncio.new_event_loop()
        try:
            from raganything.webhook_service import WebhookService
            send_loop.run_until_complete(
                WebhookService.send_webhook_with_retry(
                    webhook_url=fixed_webhook_url,
                    doc_id="",
                    document_id=document_id,
                    project_id=project_id,
                    status="failed",
                    error=error_msg,
                    max_retries=3,
                )
            )
        finally:
            send_loop.close()

    except Exception as e:
        logger.error(f"handle_job_failure: failed to send failure webhook: {e}")


def process_document_task(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Synchronous wrapper for RQ worker
    
    Args:
        task_data: Task data containing document_id, s3_url, processing_options, etc.
    
    Returns:
        Dict with processing results
    """
    # Run async function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(process_document_task_async(task_data))
    finally:
        loop.close()

