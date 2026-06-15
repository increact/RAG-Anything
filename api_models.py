"""
Pydantic request/response models for the RAG-Anything API server.

Extracted from api_server.py to keep the route handlers separate from the
wire-format schema definitions.
"""
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field


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
    vlm_model: Optional[str] = Field(None, description="Vision model used for image description, if any")


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
    lightrag_enabled: bool = Field(..., description="Whether LightRAG is enabled")
    queue_available: bool = Field(False, description="Whether the document queue (Redis) is available")


class ResultMetadata(BaseModel):
    """Metadata about a processed document"""
    parser: str = Field(..., description="Parser used (mineru or docling)", example="mineru")
    parse_method: str = Field(..., description="Parse method used (auto, ocr, txt)", example="auto")
    doc_id: str = Field(..., description="RAG-Anything document ID", example="a1b2c3d4e5f6")
    tables: int = Field(0, description="Number of tables extracted", example=3)
    formulas: int = Field(0, description="Number of formulas extracted", example=2)
    images: int = Field(0, description="Number of images extracted", example=5)
    vlm_model: Optional[str] = Field(None, description="Vision model used for image description, if any")


class DocumentResultResponse(BaseModel):
    """Full processing result returned by GET /api/v1/result/{doc_id}"""
    success: bool = Field(..., description="Whether the result was found and loaded")
    doc_id: str = Field(..., description="RAG-Anything document ID")
    document_id: str = Field("", description="Caller-supplied document identifier")
    project_id: str = Field("", description="Caller-supplied project identifier")
    processed_at: str = Field(..., description="ISO-8601 UTC timestamp of when processing completed")
    metadata: ResultMetadata = Field(..., description="Processing metadata")
    markdown: str = Field(..., description="Full Markdown content extracted from the document")
    content_list: List[Dict[str, Any]] = Field(
        ...,
        description=(
            "Structured content blocks. Each block has a 'type' field: "
            "'text', 'table', 'equation', or 'image', plus type-specific fields."
        ),
    )
