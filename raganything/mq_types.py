"""
Message Queue types for document processing tasks
"""
from typing import Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel


class DocumentProcessingTask(BaseModel):
    """Document processing task"""
    task_id: str
    document_id: str  # Client document ID
    project_id: str   # Client project ID
    webhook_url: str
    s3_url: str
    processing_options: Dict[str, Any]
    status: str = "pending"  # pending | processing | completed | webhook_sent | failed
    rag_anything_doc_id: Optional[str] = None
    webhook_attempts: int = 0
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime

