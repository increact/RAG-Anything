"""
Message Queue service for document processing tasks
Uses Redis Queue (rq) for task management
"""
import os
import json
import logging
from typing import Optional, Dict, Any
from datetime import datetime
from redis import Redis
from rq import Queue, Retry
from rq.job import Job, JobStatus

logger = logging.getLogger(__name__)


class DocumentProcessingQueue:
    """Document processing queue service"""
    
    def __init__(self):
        redis_host = os.getenv("REDIS_HOST", "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 6379))
        redis_db = int(os.getenv("REDIS_DB", 0))
        redis_password = os.getenv("REDIS_PASSWORD")
        
        self.redis_client = Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=False,  # Keep bytes for compatibility
        )
        
        self.queue = Queue("document-processing", connection=self.redis_client)
        logger.info(f"Initialized document processing queue (Redis: {redis_host}:{redis_port})")
    
    def add_task(
        self,
        task_id: str,
        document_id: str,
        project_id: str,
        webhook_url: str,
        s3_url: str,
        processing_options: Dict[str, Any],
    ) -> str:
        """
        Add task to queue
        
        Returns:
            str: Job ID
        """
        task_data = {
            "task_id": task_id,
            "document_id": document_id,
            "project_id": project_id,
            "webhook_url": webhook_url,
            "s3_url": s3_url,
            "processing_options": processing_options,
        }
        
        job = self.queue.enqueue(
            "raganything.document_worker.process_document_task",
            task_data,
            job_id=task_id,
            job_timeout="1h",  # 1 hour timeout
            result_ttl=86400,  # Keep result for 24 hours
            failure_ttl=86400,  # Keep failed jobs for 24 hours
            retry=Retry(max=3),  # Retry up to 3 times
        )
        
        logger.info(f"Added task {task_id} to queue (job_id: {job.id})")
        return job.id
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        try:
            job = Job.fetch(task_id, connection=self.redis_client)
            
            # Map RQ job status to our status
            status_map = {
                JobStatus.QUEUED: "pending",
                JobStatus.STARTED: "processing",
                JobStatus.FINISHED: "completed",
                JobStatus.FAILED: "failed",
            }
            
            status = status_map.get(job.get_status(), "unknown")
            
            # Get job data
            job_data = job.kwargs if hasattr(job, 'kwargs') else {}
            result = job.result if job.is_finished else None
            
            return {
                "status": status,
                "data": job_data,
                "result": result,
                "error": str(job.exc_info) if job.is_failed else None,
            }
        except Exception as e:
            logger.error(f"Failed to get task status for {task_id}: {e}")
            return None
    
    def get_job(self, task_id: str) -> Optional[Job]:
        """Get Job object"""
        try:
            return Job.fetch(task_id, connection=self.redis_client)
        except Exception as e:
            logger.error(f"Failed to get job for {task_id}: {e}")
            return None

