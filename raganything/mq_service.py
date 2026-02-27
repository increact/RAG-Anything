"""
Message Queue service for document processing tasks.
Uses Redis Queue (rq) for task management.
"""
import os
import logging
from typing import Optional, Dict, Any, Tuple
from redis import Redis
from rq import Queue, Retry
from rq.job import Job, JobStatus

logger = logging.getLogger(__name__)

# Redis key prefix for deduplication entries
_DEDUP_PREFIX = "rag:dedup:"


def _timeout_to_seconds(timeout) -> int:
    """
    Normalise a job-timeout value to seconds.

    Accepts:
    - int / float  → used directly
    - "7200"       → 7200
    - "2h"         → 7200
    - "30m"        → 1800
    """
    if isinstance(timeout, (int, float)):
        return int(timeout)
    s = str(timeout).strip().lower()
    try:
        if s.endswith("h"):
            return int(s[:-1]) * 3600
        if s.endswith("m"):
            return int(s[:-1]) * 60
        return int(s)
    except ValueError:
        return 7200  # safe fallback


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

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def _dedup_key(self, document_id: str) -> str:
        return f"{_DEDUP_PREFIX}{document_id}"

    def _find_active_job(self, document_id: str) -> Optional[str]:
        """
        Return the job_id of an active (queued / running) job for *document_id*,
        or None if no such job exists.

        "Active" means status is QUEUED, STARTED, or DEFERRED.  Completed
        (FINISHED / FAILED) jobs are treated as no longer active so that the
        same document can be re-submitted after processing ends.
        """
        key = self._dedup_key(document_id)
        raw = self.redis_client.get(key)
        if not raw:
            return None

        job_id = raw.decode() if isinstance(raw, bytes) else raw

        try:
            job = Job.fetch(job_id, connection=self.redis_client)
            status = job.get_status()
            if status in (JobStatus.QUEUED, JobStatus.STARTED, JobStatus.DEFERRED):
                return job_id
        except Exception:
            pass

        # Job is done, failed, or no longer exists — remove the stale key
        self.redis_client.delete(key)
        return None

    def _register_dedup(self, document_id: str, job_id: str, timeout_seconds: int) -> None:
        """
        Store the dedup entry in Redis with a TTL of timeout_seconds + 10-minute buffer.
        The buffer ensures the key outlives the job even under heavy load.
        """
        ttl = timeout_seconds + 600
        self.redis_client.setex(self._dedup_key(document_id), ttl, job_id)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_task(
        self,
        task_id: str,
        document_id: str,
        project_id: str,
        webhook_url: str,
        s3_url: str,
        processing_options: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """
        Add a document processing task to the queue.

        If an identical job for *document_id* is already active (queued or
        running), the call is a no-op and the existing job_id is returned with
        ``is_duplicate=True``.  Once the existing job finishes or fails a fresh
        submission is accepted normally.

        Returns:
            (job_id, is_duplicate)
            - job_id      — RQ job identifier (existing one if duplicate)
            - is_duplicate — True when the submission was silently skipped
        """
        # --- deduplication check ---
        active_job_id = self._find_active_job(document_id)
        if active_job_id:
            logger.warning(
                f"Document {document_id!r} is already queued / processing "
                f"(job: {active_job_id}). Skipping duplicate submission."
            )
            return active_job_id, True

        # --- build task payload ---
        task_data = {
            "task_id": task_id,
            "document_id": document_id,
            "project_id": project_id,
            "webhook_url": webhook_url,
            "s3_url": s3_url,
            "processing_options": processing_options,
        }

        raw_timeout = os.getenv("JOB_TIMEOUT", "7200")
        try:
            job_timeout: Any = int(raw_timeout)
        except ValueError:
            job_timeout = raw_timeout  # let rq parse strings like "2h"

        timeout_seconds = _timeout_to_seconds(raw_timeout)

        from raganything.document_worker import handle_job_failure

        job = self.queue.enqueue(
            "raganything.document_worker.process_document_task",
            task_data,
            job_id=task_id,
            job_timeout=job_timeout,
            result_ttl=86400,   # Keep result for 24 hours
            failure_ttl=86400,  # Keep failed jobs for 24 hours
            retry=Retry(max=3, interval=[30, 60, 120]),  # 30 s, 1 min, 2 min
            on_failure=handle_job_failure,
        )

        # Register dedup entry so re-submissions are blocked while this job runs
        self._register_dedup(document_id, job.id, timeout_seconds)

        logger.info(f"Added task {task_id} to queue (job_id: {job.id})")
        return job.id, False

    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        try:
            job = Job.fetch(task_id, connection=self.redis_client)

            status_map = {
                JobStatus.QUEUED: "pending",
                JobStatus.STARTED: "processing",
                JobStatus.FINISHED: "completed",
                JobStatus.FAILED: "failed",
            }

            status = status_map.get(job.get_status(), "unknown")
            job_data = job.kwargs if hasattr(job, "kwargs") else {}
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
