#!/usr/bin/env python3
"""
FastAPI server for RAG-Anything document processing service

This service exposes an API endpoint for processing documents and returning markdown.
Designed to be called from the tandra-ai-write backend service.
"""

import os
import sys
import asyncio
import tempfile
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Import RAG-Anything
from raganything import RAGAnything, RAGAnythingConfig
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import EmbeddingFunc

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global RAG instance (will be initialized on startup)
rag_instance: Optional[RAGAnything] = None


class ProcessDocumentRequest(BaseModel):
    """Request model for document processing"""
    parser: Optional[str] = Field(default="auto", description="Parser: auto, mineru, or docling")
    parse_method: Optional[str] = Field(default="auto", description="Parse method: auto, ocr, or txt")
    language: Optional[str] = Field(default="zh", description="Document language for OCR")
    device: Optional[str] = Field(default="cpu", description="Device: cpu, cuda:0, mps")
    formula: Optional[bool] = Field(default=True, description="Enable formula extraction")
    table: Optional[bool] = Field(default=True, description="Enable table extraction")
    backend: Optional[str] = Field(default="pipeline", description="Backend: pipeline, vlm-transformers")


class DocumentMetadata(BaseModel):
    """Metadata about the processed document"""
    parser: str = Field(..., description="Parser used (mineru or docling)")
    parse_method: str = Field(..., description="Parse method used (auto, ocr, txt)")
    doc_id: str = Field(..., description="Document ID")
    tables: int = Field(0, description="Number of tables extracted")
    formulas: int = Field(0, description="Number of formulas extracted")
    images: int = Field(0, description="Number of images extracted")


class ProcessDocumentResponse(BaseModel):
    """Response model for document processing"""
    success: bool = Field(..., description="Whether processing was successful")
    markdown: str = Field(..., description="Converted Markdown content")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")
    error: Optional[str] = Field(None, description="Error message (if processing failed)")
    debug_info: Optional[Dict[str, Any]] = Field(None, description="Debug information (development mode)")


class ProcessContentListResponse(BaseModel):
    """Response model for content_list-only processing"""
    success: bool = Field(..., description="Whether processing was successful")
    content_list: Optional[List[Dict[str, Any]]] = Field(None, description="Parsed content list")
    metadata: Optional[DocumentMetadata] = Field(None, description="Document metadata")
    error: Optional[str] = Field(None, description="Error message (if processing failed)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status", example="healthy")
    service: str = Field(..., description="Service name", example="rag-anything-api")
    rag_initialized: bool = Field(..., description="Whether RAG instance is initialized")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown"""
    global rag_instance
    
    # Startup: Initialize RAG instance
    logger.info("Initializing RAG-Anything service...")
    try:
        # Get API key from environment
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST")
        
        if not api_key:
            logger.warning("No API key found. Some features may not work.")
            api_key = "dummy-key"  # Will fail gracefully if needed
        
        # Create configuration
        config = RAGAnythingConfig(
            working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
            parser=os.getenv("PARSER", "mineru"),
            parse_method=os.getenv("PARSE_METHOD", "auto"),
            enable_image_processing=True,
            enable_table_processing=True,
            enable_equation_processing=True,
        )
        
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
        
        # Initialize RAG-Anything
        rag_instance = RAGAnything(
            config=config,
            llm_model_func=llm_model_func,
            embedding_func=embedding_func,
        )
        
        logger.info("✅ RAG-Anything service initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG-Anything: {str(e)}")
        logger.exception(e)
        # Continue anyway - will fail on first request
    
    yield
    
    # Shutdown: Cleanup
    logger.info("Shutting down RAG-Anything service...")
    rag_instance = None


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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/api/v1/doc/{doc_id}",
    tags=["Document Management"],
    summary="View document information",
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
    try:
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        output_path = Path(output_dir)
        
        # Search for files containing this doc_id
        found_files = []
        
        # Search all .md and .json files
        for md_file in output_path.rglob("*.md"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if content:
                        found_files.append({
                            "type": "markdown",
                            "path": str(md_file.relative_to(output_path)),
                            "full_path": str(md_file),
                            "size": len(content),
                        })
            except Exception:
                pass
        
        for json_file in output_path.rglob("*_content_list.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content_list = json.load(f)
                    found_files.append({
                        "type": "content_list",
                        "path": str(json_file.relative_to(output_path)),
                        "full_path": str(json_file),
                        "items": len(content_list),
                    })
            except Exception:
                pass
        
        return {
            "doc_id": doc_id,
            "found_files": found_files,
            "output_dir": output_dir,
            "note": "Use /api/v1/doc-content/{doc_id} to get actual content, or directly access files in the output directory",
        }
        
    except Exception as e:
        logger.error(f"Error retrieving document {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/doc-content/{doc_id}",
    tags=["Document Management"],
    summary="Get document Markdown content",
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
    try:
        output_dir = os.getenv("OUTPUT_DIR", "./output")
        output_path = Path(output_dir)
        
        # Search all possible markdown files
        markdown_files = list(output_path.rglob("*.md"))
        
        # Try to find the most relevant file
        markdown_content = None
        found_file = None
        
        # Sort by modification time, get the latest
        if markdown_files:
            markdown_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            latest_file = markdown_files[0]
            try:
                with open(latest_file, "r", encoding="utf-8") as f:
                    markdown_content = f.read()
                    found_file = str(latest_file.relative_to(output_path))
            except Exception as e:
                logger.warning(f"Failed to read {latest_file}: {e}")
        
        if markdown_content:
            return {
                "doc_id": doc_id,
                "markdown": markdown_content,
                "source_file": found_file,
                "length": len(markdown_content),
                "note": "Returns the most recently processed document content. If this is not what you want, please directly access files in the output directory.",
            }
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Document content not found for doc_id: {doc_id}. Please check output directory: {output_dir}",
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving document content {doc_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get(
    "/api/v1/debug/content-list/{doc_id}",
    tags=["Document Management"],
    summary="[Debug] View content_list",
    description="""
    Debug endpoint: View raw content_list for specified doc_id
    
    This endpoint returns the parsed raw content_list JSON data for debugging purposes.
    """
)
async def get_content_list(doc_id: str):
    """
    Get document raw content_list
    """
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
    """
    return HealthResponse(
        status="healthy",
        service="rag-anything-api",
        rag_initialized=rag_instance is not None,
    )


@app.post(
    "/api/v1/process",
    response_model=ProcessDocumentResponse,
    tags=["Document Processing"],
    summary="Process document and convert to Markdown",
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
    
    # Create temporary file for the uploaded file
    temp_file = None
    debug_info = {}
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
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
        
        # Update config with selected parser
        rag_instance.config.parser = selected_parser
        
        # Parse document using RAG-Anything's parse_document method
        content_list, doc_id = await rag_instance.parse_document(
            temp_file,
            output_dir,
            parse_method,
            display_stats=True,
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
        
        logger.info(f"✅ Successfully processed {file.filename}")
        logger.info(f"   - Tables: {metadata['tables']}, Formulas: {metadata['formulas']}, Images: {metadata['images']}")
        logger.info(f"   - Markdown length: {len(markdown)} chars")
        logger.info(f"   - Doc ID: {doc_id}")
        
        response_data = {
            "success": True,
            "markdown": markdown,
            "metadata": DocumentMetadata(**metadata),
        }
        
        if debug:
            response_data["debug_info"] = debug_info
        
        return ProcessDocumentResponse(**response_data)
        
    except Exception as e:
        logger.error(f"❌ Error processing document: {str(e)}")
        logger.exception(e)
        return ProcessDocumentResponse(
            success=False,
            markdown="",
            error=str(e),
            debug_info=debug_info if debug else None,
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
    
    async def generate():
        temp_file = None
        try:
            # Read file content
            file_content = await file.read()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
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
            
            # Update config with selected parser
            rag_instance.config.parser = selected_parser
            
            # Parse document using RAG-Anything
            content_list, doc_id = await rag_instance.parse_document(
                temp_file,
                output_dir,
                parse_method,
                display_stats=True,
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
    
    temp_file = None
    
    try:
        # Read file content
        file_content = await file.read()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
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
        
        # Update config with selected parser
        rag_instance.config.parser = selected_parser
        
        # Parse document using RAG-Anything's parse_document method
        content_list, doc_id = await rag_instance.parse_document(
            temp_file,
            output_dir,
            parse_method,
            display_stats=True,
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
        return ProcessContentListResponse(
            success=False,
            content_list=None,
            metadata=None,
            error=str(e),
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
