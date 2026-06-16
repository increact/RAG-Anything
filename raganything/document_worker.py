"""
Document processing worker for handling queued tasks
"""
import gc
import os
import asyncio
import tempfile
import logging
import httpx
from typing import Dict, Any
from pathlib import Path
from urllib.parse import urlparse, unquote

from fastapi import HTTPException

from raganything.image_vlm import is_image_by_content, describe_image
from validation import validate_external_url

logger = logging.getLogger(__name__)

# Global RAG instance (will be set by worker initialization)
rag_instance = None


def set_rag_instance(instance):
    """Set the global RAG instance"""
    global rag_instance
    rag_instance = instance


def _presigned_url_expired(url: str) -> bool:
    """Detect an expired AWS SigV4 presigned URL from its query string.

    AWS encodes `X-Amz-Date` + `X-Amz-Expires` in the URL; if the sum is in
    the past, the download will return 403. Returning True lets the caller
    raise a non-retryable error instead of burning two more retries on a
    download that cannot succeed. Returns False for non-S3 / non-expiring URLs.
    """
    try:
        from urllib.parse import parse_qs
        from datetime import datetime, timezone
        qs = parse_qs(urlparse(url).query)
        amz_date = qs.get("X-Amz-Date", [None])[0]
        amz_expires = qs.get("X-Amz-Expires", [None])[0]
        if not amz_date or not amz_expires:
            return False
        signed_at = datetime.strptime(amz_date, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        expires_at = signed_at.timestamp() + int(amz_expires)
        return datetime.now(timezone.utc).timestamp() >= expires_at
    except Exception:
        return False


class _NonRetryableError(Exception):
    """Marker exception that RQ retry should treat as terminal."""


def _maybe_dockerize_webhook(url: str) -> str:
    """Optionally rewrite `localhost` in a webhook URL to `host.docker.internal`.

    Only applied when DOCKER_NETWORKING=true. On bare-metal or VM deploys the
    Docker DNS name does not resolve, so an unconditional rewrite turns every
    completion webhook into a permanent silent failure.
    """
    if not url:
        return url
    if os.getenv("DOCKER_NETWORKING", "false").lower() not in ("true", "1", "yes"):
        return url
    fixed = url.replace("localhost", "host.docker.internal")
    if fixed != url:
        logger.info("Rewrote localhost -> host.docker.internal for Docker networking")
    return fixed


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

        # Defense-in-depth: re-validate s3_url at the dereference boundary.
        # api_server validates before enqueue, but any code path that lands a
        # job in Redis without going through the API (manual injection, future
        # internal callers) must not be able to make the worker fetch internal
        # endpoints — IMDS at 169.254.169.254 is the canonical concern on EC2.
        try:
            validate_external_url(s3_url, "s3_url")
        except HTTPException as e:
            raise _NonRetryableError(f"s3_url failed validation: {e.detail}") from e

        logger.info(f"Processing task {task_id} for document {document_id}")

        if rag_instance is None:
            raise Exception("RAG instance not initialized")
        
        # Extract file extension from URL path before downloading
        parsed_url = urlparse(s3_url)
        url_path = unquote(parsed_url.path)  # Decode URL encoding
        file_ext = Path(url_path).suffix or ".pdf"

        # Bail out early on an expired presigned URL so retry attempts don't
        # waste delay slots downloading a guaranteed-403 URL.
        if _presigned_url_expired(s3_url):
            raise _NonRetryableError(
                "s3_url presigned signature has expired before download"
            )

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
        language = processing_options.get("language", "ch")
        device = processing_options.get("device", "cpu")
        
        # formula/table detection each load a heavy ML model (~700MB-1GB each).
        # Default to env-var controlled values so memory usage can be tuned at
        # deployment time without changing caller code.
        # Callers can always override per-request via processing_options.
        formula_default = os.getenv("MINERU_FORMULA_DEFAULT", "false").lower() == "true"
        table_default = os.getenv("MINERU_TABLE_DEFAULT", "true").lower() == "true"

        parser_kwargs = {
            "lang": language,
            "device": device,
            "formula": processing_options.get("formula", formula_default),
            "table": processing_options.get("table", table_default),
            "backend": processing_options.get("backend", "pipeline"),
        }

        # MinerU handles all supported formats (Office docs via LibreOffice
        # preprocessing — installed in the image). The docling CLI is not
        # installed and routing Office files to it produced a 500 at runtime.
        selected_parser = "mineru" if parser == "auto" else parser
        
        logger.info(f"Parsing document with parser: {selected_parser}, method: {parse_method}")

        # Parse document — parser passed per-call to avoid mutating shared config
        content_list, doc_id = await rag_instance.parse_document(
            temp_file,
            output_dir,
            parse_method,
            display_stats=True,
            parser=selected_parser,
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

        # VLM image description — standalone image uploads only.
        vlm_model = None
        if (
            os.getenv("ENABLE_IMAGE_VLM", "true").lower() in ("true", "1", "yes")
            and is_image_by_content(temp_file)
            and getattr(rag_instance, "vision_model_func", None)
        ):
            vlm_description = await describe_image(temp_file, rag_instance.vision_model_func)
            if vlm_description:
                markdown = f"{markdown}\n\n## Image Analysis (VLM)\n\n{vlm_description}"
                # Also surface the description as a text block in content_list
                # so downstream consumers using structured indexing (which
                # ignores markdown when content_list is non-empty) still pick
                # up the VLM analysis.
                content_list.append({"type": "text", "text": vlm_description})
                vlm_model = os.getenv("VISION_MODEL", "openai/gpt-4o-mini")
                logger.info("Added VLM image description for %s", temp_file)

        # Build basic metadata counts
        metadata: dict = {
            "parser": selected_parser,
            "parse_method": parse_method,
            "doc_id": doc_id,
            "tables": 0,
            "formulas": 0,
            "images": 0,
            "vlm_model": vlm_model,
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

        # Persist result to disk so GET /api/v1/result/{doc_id} can serve it.
        # save_result now raises on disk failure — let the exception propagate
        # so the job is retried and the client is not told `completed` when no
        # result file exists.
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

        fixed_webhook_url = _maybe_dockerize_webhook(webhook_url)

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

        # Release large objects and force garbage collection to prevent
        # memory accumulation across successive jobs in this long-lived worker.
        content_list = None  # noqa: F841
        markdown = None  # noqa: F841
        gc.collect()


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

        fixed_webhook_url = _maybe_dockerize_webhook(webhook_url)

        # asyncio.run() is the safe way to drive an async coroutine from sync
        # code: it creates a fresh loop, runs the coroutine, then cleanly
        # cancels any pending tasks and closes the loop. The previous manual
        # new_event_loop / close pattern leaked tasks on the exception path.
        from raganything.webhook_service import WebhookService
        asyncio.run(
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
        # Force GC after every job to reclaim memory from MinerU models,
        # parsed content, and base64-encoded images before the next job.
        gc.collect()

