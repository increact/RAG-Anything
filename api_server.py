#!/usr/bin/env python3
"""
FastAPI server for RAG-Anything document processing service

This service exposes an API endpoint for processing documents and returning markdown.
Designed to be called from the client backend service.
"""

import os
import sys
import hmac
import asyncio
import tempfile
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import uvicorn
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, status, Security, Depends
from fastapi.security import APIKeyHeader
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from dotenv import load_dotenv

from raganything.image_vlm import build_vision_model_func, is_image_file, describe_image

from api_models import (
    DocumentMetadata,
    ProcessDocumentResponse,
    ProcessContentListResponse,
    HealthResponse,
    ResultMetadata,
    DocumentResultResponse,
)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import RAG-Anything
from raganything import RAGAnything, RAGAnythingConfig

# LightRAG imports (only needed if LightRAG is enabled)
try:
    from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    from lightrag.utils import EmbeddingFunc
    LIGHTRAG_AVAILABLE = True
except ImportError:
    LIGHTRAG_AVAILABLE = False
    logger.warning("LightRAG imports not available, RAG features will be disabled")

# Global RAG instance (will be initialized on startup)
rag_instance: Optional[RAGAnything] = None

# Semaphore to limit concurrent document processing (set in lifespan)
processing_semaphore: Optional[asyncio.Semaphore] = None

# Singleton queue service (set in lifespan)
document_queue: Optional[Any] = None

# API Key security configuration
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


# Input-validation helpers live in validation.py (standalone, unit-tested).
# Imported under the historic underscore-prefixed names used at the call sites.
from validation import (  # noqa: E402
    validate_external_url as _validate_external_url,
    validate_doc_id as _validate_doc_id,
    redact_url as _redact_url,
    safe_tempfile_suffix as _safe_tempfile_suffix,
)


async def _read_upload_bounded(file: UploadFile, max_mb: Optional[int] = None) -> bytes:
    """Read an UploadFile into memory with an enforced size cap.

    Streams the body in 1 MB chunks and aborts as soon as the running total
    exceeds the limit, so a malicious 10 GB upload does not buffer fully
    before being rejected.
    """
    if max_mb is None:
        max_mb = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    max_bytes = max_mb * 1024 * 1024
    chunks: List[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File too large. Maximum allowed size is {max_mb} MB.",
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    """
    Verify API key from request header
    
    Checks if the provided API key matches the configured API_KEY from environment.
    If API_KEY is not set in environment, authentication is disabled (development mode).
    
    Args:
        api_key: API key from X-API-Key header
        
    Raises:
        HTTPException: If API key is invalid or missing (when API_KEY is configured)
    """
    # Get configured API key from environment
    configured_api_key = os.getenv("API_KEY")
    
    # If no API key is configured, skip authentication (development mode)
    if not configured_api_key:
        logger.warning("API_KEY not configured - authentication is disabled!")
        return None
    
    # Check if API key is provided
    if not api_key:
        logger.warning("API request without API key")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is required. Please provide X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Verify API key using a constant-time comparison so an on-host attacker
    # cannot recover the key character-by-character via timing side channels.
    if not hmac.compare_digest(api_key, configured_api_key):
        logger.warning(f"Invalid API key attempt: {api_key[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key",
        )

    return api_key


async def _send_webhook(
    webhook_url: str,
    doc_id: str,
    document_id: str,
    project_id: str,
    status: str,
    error: Optional[str] = None,
):
    """Fire-and-forget webhook helper — sends a lightweight status notification only.
    Full result is available via GET /api/v1/result/{doc_id}.
    """
    from raganything.webhook_service import WebhookService
    await WebhookService.send_webhook_with_retry(
        webhook_url=webhook_url,
        doc_id=doc_id,
        document_id=document_id,
        project_id=project_id,
        status=status,
        error=error,
        max_retries=3,
    )


# asyncio.create_task() returns a task whose only strong reference is the
# caller's local variable; once that goes out of scope (which happens
# immediately for fire-and-forget patterns) the GC can collect and cancel the
# task mid-flight. Stash references in this set and drop them only when the
# task finishes.
_BACKGROUND_TASKS: set = set()


def _fire_and_forget_webhook(**kwargs) -> None:
    """Schedule a webhook send without losing the task to GC."""
    task = asyncio.create_task(_send_webhook(**kwargs))
    _BACKGROUND_TASKS.add(task)
    task.add_done_callback(_BACKGROUND_TASKS.discard)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    global rag_instance, processing_semaphore, document_queue
    
    # Initialize document worker with RAG instance
    from raganything.document_worker import set_rag_instance
    
    # Set up processing semaphore based on MAX_CONCURRENT_FILES
    max_concurrent = int(os.getenv("MAX_CONCURRENT_FILES", "1"))
    processing_semaphore = asyncio.Semaphore(max_concurrent)
    logger.info(f"Processing semaphore initialized with max_concurrent={max_concurrent}")
    
    # Check if LightRAG is enabled
    enable_lightrag = os.getenv("ENABLE_LIGHTRAG", "false").lower() in ("true", "1", "yes")
    
    # Startup: Initialize RAG instance
    if enable_lightrag:
        logger.info("Initializing RAG-Anything service with LightRAG...")
    else:
        logger.info("Initializing RAG-Anything service (LightRAG disabled, parsing only)...")
    
    try:
        # Create configuration
        config = RAGAnythingConfig(
            working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
            parser=os.getenv("PARSER", "mineru"),
            parse_method=os.getenv("PARSE_METHOD", "auto"),
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        
        # Vision model for image description (independent provider, see image_vlm).
        vision_model_func = build_vision_model_func()

        if enable_lightrag:
            if not LIGHTRAG_AVAILABLE:
                logger.error("LightRAG is enabled but imports are not available. Please install lightrag package.")
                raise ImportError("LightRAG imports not available")
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST")
            
            if not api_key:
                logger.warning("No API key found. Some features may not work.")
                api_key = "dummy-key"  # Will fail gracefully if needed
            
            # Define LLM function
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            
            # Define embedding function
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
            
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=embedding_model,
                    api_key=api_key,
                    base_url=base_url,
                ),
            )
            
            # Initialize RAG-Anything with LightRAG
            rag_instance = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
                vision_model_func=vision_model_func,
            )
            
            logger.info("✅ RAG-Anything service initialized successfully with LightRAG")
        else:
            # Initialize RAG-Anything without LightRAG (parsing only)
            # Create a dummy LightRAG instance to satisfy initialization requirements
            # but we won't use it for RAG operations
            rag_instance = RAGAnything(
                config=config,
                llm_model_func=None,
                embedding_func=None,
                vision_model_func=vision_model_func,
            )
            # Set lightrag to None explicitly to disable RAG features
            rag_instance.lightrag = None
            logger.info("✅ RAG-Anything service initialized successfully (parsing only, LightRAG disabled)")
        
        # Set RAG instance for document worker
        set_rag_instance(rag_instance)
        logger.info("✅ Document worker initialized")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG-Anything: {str(e)}")
        logger.exception(e)
        # Continue anyway - will fail on first request
    
    # Initialize queue service singleton (avoids creating a new Redis connection
    # per request inside queue_document)
    try:
        from raganything.mq_service import DocumentProcessingQueue
        document_queue = DocumentProcessingQueue()
        logger.info("✅ Document queue service initialized")
    except Exception as e:
        logger.warning(f"⚠️ Could not initialize document queue (Redis may not be running): {e}")
        document_queue = None

    # Start background output-directory cleanup task
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    ttl_days = int(os.getenv("OUTPUT_FILE_TTL_DAYS", "7"))
    interval_hours = float(os.getenv("OUTPUT_CLEANUP_INTERVAL_HOURS", "24"))

    from raganything.cleanup import run_cleanup_loop
    cleanup_task = asyncio.create_task(
        run_cleanup_loop(
            output_dir=output_dir,
            ttl_days=ttl_days,
            interval_hours=interval_hours,
            initial_delay_seconds=60.0,
        )
    )
    logger.info(
        f"✅ Output cleanup task started "
        f"(ttl={ttl_days}d, interval={interval_hours}h)"
    )

    yield

    # Shutdown: cancel the cleanup loop first, then release other resources
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

    logger.info("Shutting down RAG-Anything service...")
    rag_instance = None
    document_queue = None


# Create FastAPI app
app = FastAPI(
    title="RAG-Anything API Server",
    description="""
    RAG-Anything Document Processing API Service
    
    This service provides powerful multimodal document parsing capabilities, supporting conversion of various document formats to Markdown.
    
    ## Key Features
    
    * **Multi-format Support**: PDF, Office documents (Word, PowerPoint, Excel), images, text files
    * **Intelligent Parsing**: Automatically extracts text, tables, formulas, and images
    * **Multiple Parsers**: Supports MinerU and Docling parsing engines
    * **GPU Acceleration**: Optional GPU acceleration to improve OCR and table extraction speed
    * **Streaming Processing**: Supports real-time streaming of processing results
    
    ## Supported Document Formats
    
    * **PDF**: Full support, including OCR, table, and formula extraction
    * **Office Documents**: DOC, DOCX, PPT, PPTX, XLS, XLSX (requires LibreOffice)
    * **Images**: JPG, PNG, BMP, TIFF, GIF, WebP (supports OCR)
    * **Text**: TXT, MD
    
    ## Usage Instructions
    
    1. Upload document to `/api/v1/process` endpoint
    2. Select parser and processing options
    3. Get converted Markdown content
    
    For more information, please refer to: https://github.com/HKUDS/RAG-Anything
    """,
    version="1.0.0",
    lifespan=lifespan,
    contact={
        "name": "RAG-Anything",
        "url": "https://github.com/HKUDS/RAG-Anything",
        "email": None,
    },
    license_info={
        "name": "MIT",
        "url": "https://github.com/HKUDS/RAG-Anything/blob/main/LICENSE",
    },
    servers=[
        {
            "url": "http://localhost:8000",
            "description": "Local development server",
        },
        {
            "url": "https://api.example.com",
            "description": "Production server (example)",
        },
    ],
)

# Configure CORS.
# When CORS_ALLOW_ORIGINS is unset we fall back to "*" + allow_credentials=False
# (the only spec-legal combination) so the API key auth path keeps working from
# arbitrary non-browser clients without silently breaking browser preflights.
_cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "*").strip()
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
_cors_allow_credentials = "*" not in _cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=_cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/api/v1/doc/{doc_id}",
    tags=["Document Management"],
    summary="View document information",
    dependencies=[Depends(verify_api_key)],
    description="""
    View processed document information by doc_id
    
    Returns basic document information and related file paths.
    """,
    responses={
        200: {
            "description": "Document information",
        },
        404: {
            "description": "Document not found",
        },
    },
)
async def get_document_by_id(doc_id: str):
    """
    Get document information by doc_id
    """
    _validate_doc_id(doc_id)
    from raganything.result_store import load_result
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    record = load_result(output_dir, doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")
    return {
        "doc_id": record["doc_id"],
        "document_id": record.get("document_id", ""),
        "project_id": record.get("project_id", ""),
        "processed_at": record.get("processed_at", ""),
        "metadata": record.get("metadata", {}),
        "markdown_size": len(record.get("markdown", "")),
        "content_blocks": len(record.get("content_list", [])),
    }


@app.get(
    "/api/v1/doc-content/{doc_id}",
    tags=["Document Management"],
    summary="Get document Markdown content",
    dependencies=[Depends(verify_api_key)],
    description="""
    Get complete Markdown content of a document by doc_id
    
    This endpoint searches the output directory, finds the Markdown file related to the doc_id, and returns its content.
    If not found, it returns the most recently processed document content.
    """,
)
async def get_document_content(doc_id: str):
    """
    Get document Markdown content
    """
    _validate_doc_id(doc_id)
    from raganything.result_store import load_result
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    record = load_result(output_dir, doc_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Document content not found for doc_id: {doc_id}",
        )
    markdown = record.get("markdown", "")
    return {
        "doc_id": doc_id,
        "markdown": markdown,
        "length": len(markdown),
    }


@app.get(
    "/api/v1/debug/content-list/{doc_id}",
    tags=["Document Management"],
    summary="[Debug] View content_list",
    dependencies=[Depends(verify_api_key)],
    description="""
    Debug endpoint: View raw content_list for specified doc_id
    
    This endpoint returns the parsed raw content_list JSON data for debugging purposes.
    """
)
async def get_content_list(doc_id: str):
    """
    Get document raw content_list
    """
    _validate_doc_id(doc_id)
    try:
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        output_path = Path(output_dir)
        
        # Search for content_list JSON files
        json_files = list(output_path.rglob("*_content_list.json"))
        
        if not json_files:
            raise HTTPException(
                status_code=404,
                detail=f"No content_list files found in {output_dir}",
            )
        
        # Sort by modification time, get the latest
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_file = json_files[0]
        
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                content_list = json.load(f)
            
            # Analyze content_list
            content_types = {}
            for item in content_list:
                if isinstance(item, dict):
                    content_type = item.get("type", "unknown")
                    content_types[content_type] = content_types.get(content_type, 0) + 1
            
            return {
                "doc_id": doc_id,
                "source_file": str(latest_file.relative_to(output_path)),
                "total_items": len(content_list),
                "content_types": content_types,
                "content_list": content_list[:10],  # Only return first 10 items
                "note": "Only showing first 10 items. For full content, please directly access the JSON file.",
            }
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read content_list file: {str(e)}",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving content_list {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/middle-json/{doc_id}",
    tags=["Document Management"],
    summary="Get document middle.json",
    dependencies=[Depends(verify_api_key)],
    description="""
    Get middle.json file for a document by doc_id
    
    This endpoint searches the output directory, finds the middle.json file related to the doc_id, and returns its content.
    The middle.json contains detailed layout information including preproc_blocks, lines, spans, etc.
    """,
)
async def get_middle_json(doc_id: str):
    """
    Get document middle.json content
    """
    _validate_doc_id(doc_id)
    try:
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        output_path = Path(output_dir)
        
        # Search for middle.json files
        json_files = list(output_path.rglob("*_middle.json"))
        
        if not json_files:
            raise HTTPException(
                status_code=404,
                detail=f"No middle.json files found in {output_dir}",
            )
        
        # Sort by modification time, get the latest
        json_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        latest_file = json_files[0]
        
        try:
            with open(latest_file, "r", encoding="utf-8") as f:
                middle_json = json.load(f)
            
            return {
                "doc_id": doc_id,
                "source_file": str(latest_file.relative_to(output_path)),
                "middle_json": middle_json,
            }
        except json.JSONDecodeError as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to parse middle.json file: {str(e)}",
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to read middle.json file: {str(e)}",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving middle.json {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"],
    summary="Health check",
    description="Check if the service is running normally and if the RAG instance is initialized",
    responses={
        200: {
            "description": "Service running normally",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "service": "rag-anything-api",
                        "rag_initialized": True,
                    }
                }
            },
        }
    },
)
async def health_check():
    """
    Health check endpoint
    
    Returns the health status of the service, including:
    - Service status
    - Service name
    - RAG instance initialization status
    - LightRAG enabled status
    """
    enable_lightrag = os.getenv("ENABLE_LIGHTRAG", "false").lower() in ("true", "1", "yes")
    return HealthResponse(
        status="healthy",
        service="rag-anything-api",
        rag_initialized=rag_instance is not None,
        lightrag_enabled=enable_lightrag and (rag_instance is not None and rag_instance.lightrag is not None),
        queue_available=document_queue is not None,
    )


# ---------------------------------------------------------------------------
# Result retrieval — called by the webhook receiver after notification
# ---------------------------------------------------------------------------

@app.get(
    "/api/v1/result/{doc_id}",
    response_model=DocumentResultResponse,
    tags=["Result Retrieval"],
    summary="Fetch full processing result by doc_id",
    dependencies=[Depends(verify_api_key)],
    description="""
Retrieve the complete processing result for a document identified by its **doc_id**.

## Intended usage

This endpoint is designed to be called by the **webhook receiver** after it receives
a completion notification. The notification webhook only carries a lightweight payload:

```json
{
  "docId": "a1b2c3...",
  "documentId": "your-doc-id",
  "projectId": "your-project-id",
  "status": "completed",
  "timestamp": "2026-02-27T12:00:00Z"
}
```

The receiver should then call this endpoint to obtain the full result.

## Result persistence

Results are stored on disk as `{output_dir}/{doc_id}_result.json` and are available
immediately after the webhook is sent. They persist until manually cleaned up or until
the optional output-directory TTL job removes them.

## doc_id

The `doc_id` is a **content-based hash** generated by RAG-Anything after parsing.
It is included in:
- The webhook notification (`docId` field)
- The synchronous `/api/v1/process` response (`metadata.doc_id`)
- The queue endpoint's job result

## Error codes

| Code | Meaning |
|------|---------|
| 200  | Result found and returned |
| 404  | No result found for this doc_id (not yet processed, or already cleaned up) |
| 503  | Service not initialised |
""",
    responses={
        200: {
            "description": "Result found",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "doc_id": "a1b2c3d4e5f6789012345678",
                        "document_id": "my-document-001",
                        "project_id": "my-project",
                        "processed_at": "2026-02-27T12:00:00+00:00",
                        "metadata": {
                            "parser": "mineru",
                            "parse_method": "auto",
                            "doc_id": "a1b2c3d4e5f6789012345678",
                            "tables": 3,
                            "formulas": 2,
                            "images": 5,
                        },
                        "markdown": "# Document Title\n\nFirst paragraph…",
                        "content_list": [
                            {"type": "text", "text": "Document Title", "text_level": 1, "page_idx": 0},
                            {"type": "table", "table_body": "| Col1 | Col2 |\n|---|---|", "page_idx": 1},
                        ],
                    }
                }
            },
        },
        404: {
            "description": "Result not found",
            "content": {
                "application/json": {
                    "example": {"detail": "Result not found for doc_id: a1b2c3d4. The document may not have been processed yet, or the result has expired."}
                }
            },
        },
    },
)
async def get_result(doc_id: str):
    """
    Return the full processing result (markdown + content_list) for the given doc_id.

    Call this endpoint from your webhook receiver after receiving a completion notification.
    """
    _validate_doc_id(doc_id)
    output_dir = os.getenv("OUTPUT_DIR", "./output")

    from raganything.result_store import load_result
    record = load_result(output_dir, doc_id)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Result not found for doc_id: {doc_id}. "
                "The document may not have been processed yet, or the result has expired."
            ),
        )

    try:
        return DocumentResultResponse(
            success=True,
            doc_id=record["doc_id"],
            document_id=record.get("document_id", ""),
            project_id=record.get("project_id", ""),
            processed_at=record.get("processed_at", ""),
            metadata=ResultMetadata(**record["metadata"]),
            markdown=record.get("markdown", ""),
            content_list=record.get("content_list", []),
        )
    except Exception as e:
        logger.error(f"Failed to deserialise result for doc_id={doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load result: {e}")


@app.post(
    "/api/v1/queue/document",
    tags=["Document Processing"],
    summary="Queue document for processing",
    dependencies=[Depends(verify_api_key)],
    description="""
    Queue a document for asynchronous processing.
    
    This endpoint accepts a document task and adds it to the processing queue.
    The document will be processed by a background worker, and a webhook will be
    sent to the specified URL when processing is complete.
    """,
)
async def queue_document(
    document_id: str = Form(..., description="Client document ID"),
    project_id: str = Form(..., description="Client project ID"),
    webhook_url: str = Form(..., description="Webhook URL to call when processing is complete"),
    s3_url: str = Form(..., description="S3 URL of the document file"),
    processing_options: str = Form(..., description="JSON string of processing options"),
):
    """Add document task to processing queue"""
    from datetime import datetime
    
    if rag_instance is None:
        raise HTTPException(
            status_code=503,
            detail="RAG-Anything service not initialized. Please check server logs."
        )
    
    if document_queue is None:
        raise HTTPException(
            status_code=503,
            detail="Document queue not available. Please check Redis connection.",
        )

    _validate_external_url(webhook_url, "webhook_url")
    _validate_external_url(s3_url, "s3_url")
    _validate_doc_id(document_id)
    _validate_doc_id(project_id)

    try:
        # Parse processing options
        options = json.loads(processing_options)
        task_id = f"task-{document_id}-{int(datetime.now().timestamp() * 1000)}"
        
        # Add task to queue (returns existing job_id when duplicate is detected)
        job_id, is_duplicate = document_queue.add_task(
            task_id=task_id,
            document_id=document_id,
            project_id=project_id,
            webhook_url=webhook_url,
            s3_url=s3_url,
            processing_options=options,
        )
        
        if is_duplicate:
            logger.info(
                f"Duplicate submission ignored for document {document_id} "
                f"(existing job: {job_id})"
            )
            return {
                "success": True,
                "task_id": job_id,
                "job_id": job_id,
                "duplicate": True,
                "message": "Document is already queued or being processed. "
                           "A webhook will be sent when the existing job completes.",
            }

        logger.info(f"Queued document {document_id} for processing (task_id: {task_id}, job_id: {job_id})")
        
        return {
            "success": True,
            "task_id": task_id,
            "job_id": job_id,
            "duplicate": False,
            "message": "Document queued for processing",
        }
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in processing_options: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=f"Invalid JSON in processing_options: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Failed to queue document: {str(e)}")
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue document: {str(e)}"
        )


@app.post(
    "/api/v1/process",
    response_model=ProcessDocumentResponse,
    tags=["Document Processing"],
    summary="Process document and convert to Markdown",
    dependencies=[Depends(verify_api_key)],
    description="""
    Upload document and convert to Markdown format
    
    This endpoint receives file uploads, processes documents using RAG-Anything, and returns extracted Markdown content.
    Supports multiple document formats including PDF, Office documents, images, etc.
    
    ## Processing Flow
    
    1. Receive uploaded file
    2. Select parser based on file type (MinerU or Docling)
    3. Extract text, tables, formulas, images, etc.
    4. Convert to Markdown format
    5. Return processing results and metadata
    
    ## Parameter Description
    
    - **parser**: Parser selection
      - `auto`: Automatically select based on file type (recommended)
      - `mineru`: Use MinerU parser (suitable for PDF, images)
      - `docling`: Use Docling parser (suitable for Office documents)
    
    - **parse_method**: Parse method
      - `auto`: Automatically select best method
      - `ocr`: Force OCR usage (suitable for scanned documents)
      - `txt`: Extract text only (fastest)
    
    - **language**: Document language (for OCR optimization)
      - `zh`: Chinese
      - `en`: English
      - Other language codes
    
    - **device**: Processing device
      - `cpu`: CPU processing (default)
      - `cuda:0`: GPU acceleration (requires NVIDIA GPU)
      - `mps`: Apple Silicon GPU
    
    - **formula**: Whether to extract formulas (default: true)
    - **table**: Whether to extract tables (default: true)
    """,
    responses={
        200: {
            "description": "Processing successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "markdown": "# Document Title\n\nThis is the extracted content...\n\n| Table | Data |\n|-------|------|\n| ... | ... |\n\n$$E = mc^2$$",
                        "metadata": {
                            "parser": "mineru",
                            "parse_method": "auto",
                            "doc_id": "doc_123456",
                            "tables": 3,
                            "formulas": 5,
                            "images": 2,
                        },
                        "error": None,
                    }
                }
            },
        },
        400: {
            "description": "Invalid request parameters",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid file format or missing file"
                    }
                }
            },
        },
        503: {
            "description": "Service not initialized",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "RAG-Anything service not initialized. Please check server logs."
                    }
                }
            },
        },
    },
)
async def process_document(
    file: UploadFile = File(
        ...,
        description="Document file to process (supports PDF, Office documents, images, etc.)",
        example="document.pdf",
    ),
    parser: str = Form(
        "auto",
        description="Parser selection: auto (automatic), mineru, docling",
        example="auto",
    ),
    parse_method: str = Form(
        "auto",
        description="Parse method: auto (automatic), ocr, txt",
        example="auto",
    ),
    language: str = Form(
        "zh",
        description="Document language (for OCR optimization): zh (Chinese), en (English), etc.",
        example="zh",
    ),
    device: str = Form(
        "cpu",
        description="Processing device: cpu, cuda:0 (GPU), mps (Apple Silicon)",
        example="cpu",
    ),
    formula: bool = Form(
        True,
        description="Whether to extract formulas",
        example=True,
    ),
    table: bool = Form(
        True,
        description="Whether to extract tables",
        example=True,
    ),
    backend: str = Form(
        "pipeline",
        description="Backend type: pipeline, vlm-transformers",
        example="pipeline",
    ),
    debug: bool = Form(
        False,
        description="Whether to return debug information",
        example=False,
    ),
    webhook_url: Optional[str] = Form(
        None,
        description="Webhook URL to call when processing is complete (optional)",
        example="http://localhost:3000/api/webhooks/rag-anything/callback",
    ),
    document_id: Optional[str] = Form(
        None,
        description="Client document ID (required if webhook_url is provided)",
        example="doc-123",
    ),
    project_id: Optional[str] = Form(
        None,
        description="Client project ID (required if webhook_url is provided)",
        example="project-123",
    ),
    session_id: Optional[str] = Form(
        None,
        description="Session ID for document processing (optional)",
        example="session-123",
    ),
):
    """
    Process document and return Markdown
    
    This endpoint receives file uploads, processes documents using RAG-Anything, and returns extracted Markdown content.
    """
    if rag_instance is None:
        raise HTTPException(
            status_code=503,
            detail="RAG-Anything service not initialized. Please check server logs."
        )

    if webhook_url:
        _validate_external_url(webhook_url, "webhook_url")
    if document_id:
        _validate_doc_id(document_id)
    if project_id:
        _validate_doc_id(project_id)
    suffix = _safe_tempfile_suffix(file.filename)

    # Create temporary file for the uploaded file
    temp_file = None
    debug_info = {}

    try:
        # Read file content with bounded size (streams + aborts on overflow).
        file_content = await _read_upload_bounded(file)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            temp_file = tmp.name

        logger.info(f"Processing file: {file.filename} (size: {len(file_content)} bytes)")
        
        # Determine parser
        if parser == "auto":
            # Auto-select based on file type
            file_ext = Path(file.filename).suffix.lower()
            if file_ext in [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]:
                selected_parser = "docling"
            else:
                selected_parser = "mineru"
        else:
            selected_parser = parser
        
        # Prepare parser kwargs
        parser_kwargs = {
            "lang": language,
            "device": device,
            "formula": formula,
            "table": table,
            "backend": backend,
        }
        
        # Process document using RAG-Anything
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Acquire semaphore to honour MAX_CONCURRENT_FILES. Pass the per-request
        # parser through explicitly instead of mutating shared config state.
        async with processing_semaphore:
            content_list, doc_id = await rag_instance.parse_document(
                temp_file,
                output_dir,
                parse_method,
                display_stats=True,
                parser=selected_parser,
                **parser_kwargs,
            )
        
        logger.info(f"Parsed document: {len(content_list)} content blocks, doc_id: {doc_id}")
        
        # Debug: Log content_list structure
        if content_list:
            first_item = content_list[0] if content_list else {}
            logger.info(f"First content item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}")
            logger.info(f"Content types: {[item.get('type', 'unknown') if isinstance(item, dict) else type(item).__name__ for item in content_list[:5]]}")
        
        if debug:
            debug_info["content_list_length"] = len(content_list)
            debug_info["content_list_sample"] = content_list[:3] if content_list else []
            debug_info["content_types"] = {}
            for item in content_list:
                if isinstance(item, dict):
                    content_type = item.get("type", "unknown")
                    debug_info["content_types"][content_type] = debug_info["content_types"].get(content_type, 0) + 1
        
        # Try to read markdown file from output directory
        markdown_from_file = ""
        file_stem = Path(temp_file).stem
        md_file = Path(output_dir) / f"{file_stem}.md"
        
        # Check for nested structure (mineru output)
        if not md_file.exists():
            file_stem_subdir = Path(output_dir) / file_stem
            if file_stem_subdir.exists():
                md_file = file_stem_subdir / parse_method / f"{file_stem}.md"
                if not md_file.exists() and parse_method == "auto":
                    # Try vlm or other methods
                    for method in ["vlm", "ocr", "txt"]:
                        potential_md = file_stem_subdir / method / f"{file_stem}.md"
                        if potential_md.exists():
                            md_file = potential_md
                            break
        
        if md_file.exists():
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    markdown_from_file = f.read()
                logger.info(f"Read markdown from file: {md_file} ({len(markdown_from_file)} chars)")
                if debug:
                    debug_info["markdown_source"] = "file"
                    debug_info["markdown_file"] = str(md_file)
            except Exception as e:
                logger.warning(f"Failed to read markdown file {md_file}: {e}")
        
        # Convert content list to markdown (fallback if file doesn't exist)
        markdown_parts = []
        metadata = {
            "parser": selected_parser,
            "parse_method": parse_method,
            "doc_id": doc_id,
            "tables": 0,
            "formulas": 0,
            "images": 0,
            "vlm_model": None,
        }
        
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                
                if content_type == "text":
                    # Text content uses "text" field
                    text_content = item.get("text", "")
                    if text_content:
                        markdown_parts.append(text_content)
                elif content_type == "table":
                    # Table content uses "table_body" field
                    table_body = item.get("table_body", "")
                    if table_body:
                        # Add caption if available
                        table_caption = item.get("table_caption", [])
                        if table_caption:
                            markdown_parts.append(f"\n\n**{table_caption[0]}**\n\n")
                        markdown_parts.append(f"{table_body}\n\n")
                    metadata["tables"] += 1
                elif content_type == "equation":
                    # Equation content uses "latex" field
                    formula_latex = item.get("latex", "")
                    if formula_latex:
                        markdown_parts.append(f"\n\n$${formula_latex}$$\n\n")
                    metadata["formulas"] += 1
                elif content_type == "image":
                    # Image content
                    img_path = item.get("img_path", "")
                    image_caption = item.get("image_caption", [])
                    caption_text = image_caption[0] if image_caption else "Image"
                    if img_path:
                        markdown_parts.append(f"\n\n![{caption_text}]({img_path})\n\n")
                    metadata["images"] += 1
        
        # Use markdown from file if available, otherwise use converted content
        if markdown_from_file:
            markdown = markdown_from_file
            logger.info("Using markdown from parsed file")
        else:
            markdown = "\n".join(markdown_parts).strip()
            logger.info(f"Using converted markdown from content_list ({len(markdown)} chars)")
            if debug:
                debug_info["markdown_source"] = "content_list"
                debug_info["markdown_parts_count"] = len(markdown_parts)
        
        # If still empty, log warning
        if not markdown:
            logger.warning(f"Markdown is empty! Content list has {len(content_list)} items")
            logger.warning(f"Content list sample: {content_list[:2] if content_list else 'Empty'}")
            if debug:
                debug_info["warning"] = "Markdown is empty"
                debug_info["content_list_sample"] = content_list[:5] if content_list else []
        
        # VLM image description — standalone image uploads only. Runs after
        # parsing; failures are swallowed (describe_image returns "").
        if (
            os.getenv("ENABLE_IMAGE_VLM", "true").lower() in ("true", "1", "yes")
            and is_image_file(file.filename)
            and getattr(rag_instance, "vision_model_func", None)
        ):
            vlm_description = await describe_image(temp_file, rag_instance.vision_model_func)
            if vlm_description:
                markdown = f"{markdown}\n\n## Image Analysis (VLM)\n\n{vlm_description}"
                metadata["vlm_model"] = os.getenv("VISION_MODEL", "openai/gpt-4o-mini")
                logger.info("Added VLM image description for %s", file.filename)

        logger.info(f"✅ Successfully processed {file.filename}")
        logger.info(f"   - Tables: {metadata['tables']}, Formulas: {metadata['formulas']}, Images: {metadata['images']}")
        logger.info(f"   - Markdown length: {len(markdown)} chars")
        logger.info(f"   - Doc ID: {doc_id}")

        # Persist full result so GET /api/v1/result/{doc_id} can serve it
        from raganything.result_store import save_result
        save_result(
            output_dir=output_dir,
            doc_id=doc_id,
            document_id=document_id or "",
            project_id=project_id or "",
            markdown=markdown,
            content_list=content_list,
            metadata=metadata,
        )
        
        # Send lightweight notification webhook (no content payload)
        if webhook_url:
            if not document_id or not project_id:
                logger.warning("webhook_url provided but document_id or project_id is missing. Skipping webhook.")
            else:
                _fire_and_forget_webhook(
                    webhook_url=webhook_url,
                    doc_id=doc_id,
                    document_id=document_id,
                    project_id=project_id,
                    status="completed",
                )
                logger.info(f"Webhook notification scheduled for document: {document_id}")
        
        response_data = {
            "success": True,
            "markdown": markdown,
            "metadata": DocumentMetadata(**metadata),
        }
        
        if debug:
            response_data["debug_info"] = debug_info
        
        return ProcessDocumentResponse(**response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing document: {str(e)}")
        logger.exception(e)

        # Send failure notification webhook
        if webhook_url and document_id and project_id:
            _fire_and_forget_webhook(
                webhook_url=webhook_url,
                doc_id="",
                document_id=document_id,
                project_id=project_id,
                status="failed",
                error=str(e),
            )
            logger.info(f"Failure webhook notification scheduled for document: {document_id}")

        # Return 500 with the same response shape so callers checking either
        # HTTP status or success=False both see the failure.
        return JSONResponse(
            status_code=500,
            content=ProcessDocumentResponse(
                success=False,
                markdown="",
                error=str(e),
                debug_info=debug_info if debug else None,
            ).model_dump(),
        )
    
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


@app.post(
    "/api/v1/process-stream",
    tags=["Document Processing"],
    summary="Stream document processing",
    dependencies=[Depends(verify_api_key)],
    description="""
    Stream document processing and return Markdown fragments in real-time
    
    This endpoint uses Server-Sent Events (SSE) to return processing results in real-time, suitable for large documents and scenarios requiring real-time feedback.
    
    ## Response Format
    
    Returns a stream in `text/event-stream` format, each data chunk starts with `data: `:
    
    ```
    data: First segment content

    data: Second segment content

    data: | Table | Data |
    |------|------|
    | ... | ... |

    data: $$Formula content$$

    data: [DONE]
    ```
    
    ## Error Handling
    
    If an error occurs during processing, it returns:
    ```
    data: [ERROR] Error message
    ```
    """,
    responses={
        200: {
            "description": "Streaming response",
            "content": {
                "text/event-stream": {
                    "example": "data: Content fragment\n\ndata: [DONE]\n\n",
                }
            },
        },
        503: {
            "description": "Service not initialized",
        },
    },
)
async def process_document_stream(
    file: UploadFile = File(
        ...,
        description="Document file to process",
    ),
    parser: str = Form(
        "auto",
        description="Parser selection",
    ),
    parse_method: str = Form(
        "auto",
        description="Parse method",
    ),
    language: str = Form(
        "zh",
        description="Document language",
    ),
    device: str = Form(
        "cpu",
        description="Processing device",
    ),
    formula: bool = Form(
        True,
        description="Whether to extract formulas",
    ),
    table: bool = Form(
        True,
        description="Whether to extract tables",
    ),
    backend: str = Form(
        "pipeline",
        description="Backend type",
    ),
):
    """
    Stream document processing and return Markdown fragments in real-time
    
    Returns Server-Sent Events (SSE) stream.
    """
    if rag_instance is None:
        raise HTTPException(
            status_code=503,
            detail="RAG-Anything service not initialized"
        )
    
    suffix = _safe_tempfile_suffix(file.filename)

    async def generate():
        temp_file = None
        try:
            # Read file content with bounded size (streams + aborts on overflow).
            file_content = await _read_upload_bounded(file)

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(file_content)
                temp_file = tmp.name
            
            # Process and yield chunks
            output_dir = os.getenv("OUTPUT_DIR", "./output")
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine parser
            if parser == "auto":
                file_ext = Path(file.filename).suffix.lower()
                selected_parser = "docling" if file_ext in [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"] else "mineru"
            else:
                selected_parser = parser
            
            parser_kwargs = {
                "lang": language,
                "device": device,
                "formula": formula,
                "table": table,
                "backend": backend,
            }
            
            # Acquire semaphore to honour MAX_CONCURRENT_FILES. Per-request parser
            # passed explicitly instead of mutating shared config.
            async with processing_semaphore:
                content_list, doc_id = await rag_instance.parse_document(
                    temp_file,
                    output_dir,
                    parse_method,
                    display_stats=True,
                    parser=selected_parser,
                    **parser_kwargs,
                )
            
            # Try to read markdown file first
            file_stem = Path(temp_file).stem
            md_file = Path(output_dir) / f"{file_stem}.md"
            
            if not md_file.exists():
                file_stem_subdir = Path(output_dir) / file_stem
                if file_stem_subdir.exists():
                    md_file = file_stem_subdir / parse_method / f"{file_stem}.md"
            
            if md_file.exists():
                # Stream from file
                with open(md_file, "r", encoding="utf-8") as f:
                    chunk_size = 1024
                    while True:
                        chunk = f.read(chunk_size)
                        if not chunk:
                            break
                        yield f"data: {chunk}\n\n"
            else:
                # Stream from content_list
                for item in content_list:
                    if isinstance(item, dict):
                        content_type = item.get("type", "text")
                        
                        if content_type == "text":
                            text_content = item.get("text", "")
                            if text_content:
                                yield f"data: {text_content}\n\n"
                        elif content_type == "table":
                            table_body = item.get("table_body", "")
                            if table_body:
                                yield f"data: \n\n{table_body}\n\n\n\n"
                        elif content_type == "equation":
                            formula_latex = item.get("latex", "")
                            if formula_latex:
                                yield f"data: \n\n$${formula_latex}$$\n\n\n\n"
            
            yield "data: [DONE]\n\n"
            
            # Cleanup
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
                
        except Exception as e:
            logger.error(f"Stream processing error: {str(e)}")
            yield f"data: [ERROR] {str(e)}\n\n"
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except:
                    pass
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post(
    "/api/v1/process-content-list",
    response_model=ProcessContentListResponse,
    tags=["Document Processing"],
    summary="Process document and return content_list only (no markdown)",
    dependencies=[Depends(verify_api_key)],
    description="""
    Upload document and return only content_list without generating markdown
    
    This endpoint is optimized for structured indexing scenarios where markdown is not needed.
    It processes documents using RAG-Anything and returns only the structured content_list.
    
    ## Use Cases
    
    - Pure indexing scenarios (no user viewing/editing needed)
    - Batch processing large volumes of documents
    - Scenarios requiring maximum processing speed
    
    ## Response Format
    
    Returns only content_list and metadata, skipping markdown generation entirely.
    """,
    responses={
        200: {
            "description": "Processing successful",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "content_list": [
                            {
                                "type": "text",
                                "text": "Document content...",
                                "text_level": 1,
                                "page_idx": 0,
                            },
                            {
                                "type": "table",
                                "table_body": "| Col1 | Col2 |\n|------|------|",
                                "table_caption": ["Table 1"],
                            },
                        ],
                        "metadata": {
                            "parser": "mineru",
                            "parse_method": "auto",
                            "doc_id": "doc_123456",
                            "tables": 3,
                            "formulas": 5,
                            "images": 2,
                        },
                        "error": None,
                    }
                }
            },
        },
        400: {
            "description": "Invalid request parameters",
        },
        503: {
            "description": "Service not initialized",
        },
    },
)
async def process_document_content_list(
    file: UploadFile = File(
        ...,
        description="Document file to process",
        example="document.pdf",
    ),
    parser: str = Form(
        "auto",
        description="Parser selection: auto (automatic), mineru, docling",
        example="auto",
    ),
    parse_method: str = Form(
        "auto",
        description="Parse method: auto (automatic), ocr, txt",
        example="auto",
    ),
    language: str = Form(
        "zh",
        description="Document language (for OCR optimization): zh (Chinese), en (English), etc.",
        example="zh",
    ),
    device: str = Form(
        "cpu",
        description="Processing device: cpu, cuda:0 (GPU), mps (Apple Silicon)",
        example="cpu",
    ),
    formula: bool = Form(
        True,
        description="Whether to extract formulas",
        example=True,
    ),
    table: bool = Form(
        True,
        description="Whether to extract tables",
        example=True,
    ),
    backend: str = Form(
        "pipeline",
        description="Backend type: pipeline, vlm-transformers",
        example="pipeline",
    ),
):
    """
    Process document and return only content_list (no markdown generation)
    
    Optimized endpoint for structured indexing scenarios.
    """
    if rag_instance is None:
        raise HTTPException(
            status_code=503,
            detail="RAG-Anything service not initialized. Please check server logs."
        )
    
    suffix = _safe_tempfile_suffix(file.filename)
    temp_file = None

    try:
        # Read file content with bounded size (streams + aborts on overflow).
        file_content = await _read_upload_bounded(file)

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(file_content)
            temp_file = tmp.name

        logger.info(f"Processing file (content_list only): {file.filename} (size: {len(file_content)} bytes)")
        
        # Determine parser
        if parser == "auto":
            file_ext = Path(file.filename).suffix.lower()
            if file_ext in [".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"]:
                selected_parser = "docling"
            else:
                selected_parser = "mineru"
        else:
            selected_parser = parser
        
        # Prepare parser kwargs
        parser_kwargs = {
            "lang": language,
            "device": device,
            "formula": formula,
            "table": table,
            "backend": backend,
        }
        
        # Process document using RAG-Anything
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Acquire semaphore + pass per-request parser through; do not mutate config.
        async with processing_semaphore:
            content_list, doc_id = await rag_instance.parse_document(
                temp_file,
                output_dir,
                parse_method,
                display_stats=True,
                parser=selected_parser,
                **parser_kwargs,
            )
        
        logger.info(f"Parsed document: {len(content_list)} content blocks, doc_id: {doc_id}")
        
        # Calculate metadata from content_list
        metadata = {
            "parser": selected_parser,
            "parse_method": parse_method,
            "doc_id": doc_id,
            "tables": 0,
            "formulas": 0,
            "images": 0,
        }
        
        for item in content_list:
            if isinstance(item, dict):
                content_type = item.get("type", "text")
                if content_type == "table":
                    metadata["tables"] += 1
                elif content_type == "equation":
                    metadata["formulas"] += 1
                elif content_type == "image":
                    metadata["images"] += 1
        
        if not content_list or len(content_list) == 0:
            raise HTTPException(
                status_code=400,
                detail="Failed to extract content from document. The document may be empty or unsupported."
            )
        
        logger.info(f"✅ Successfully processed {file.filename} (content_list only)")
        logger.info(f"   - Tables: {metadata['tables']}, Formulas: {metadata['formulas']}, Images: {metadata['images']}")
        logger.info(f"   - Content list items: {len(content_list)}")
        logger.info(f"   - Doc ID: {doc_id}")
        
        return ProcessContentListResponse(
            success=True,
            content_list=content_list,
            metadata=DocumentMetadata(**metadata),
            error=None,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing document (content_list only): {str(e)}")
        logger.exception(e)
        return JSONResponse(
            status_code=500,
            content=ProcessContentListResponse(
                success=False,
                content_list=None,
                metadata=None,
                error=str(e),
            ).model_dump(),
        )
    
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except Exception as e:
                logger.warning(f"Failed to delete temp file: {e}")


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    logger.info(f"Starting RAG-Anything API server on {host}:{port}")
    
    uvicorn.run(
        "api_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )
