"""
Webhook service with retry mechanism for sending callbacks to client.

Design: webhooks carry only a lightweight notification (status + IDs).
Full processing results are served via GET /api/v1/result/{doc_id}.
"""
import asyncio
import httpx
from datetime import datetime, timezone
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class WebhookService:
    """Webhook sending service (with retry mechanism)"""

    @staticmethod
    async def send_webhook_with_retry(
        webhook_url: str,
        doc_id: str,
        document_id: str,
        project_id: str,
        status: str,
        error: Optional[str] = None,
        max_retries: int = 3,
    ) -> bool:
        """
        Send a lightweight status-notification webhook (with exponential-backoff retry).

        The payload intentionally omits the processed content (markdown / content_list).
        Receivers should call GET /api/v1/result/{doc_id} to fetch the full result.

        Args:
            webhook_url:  Destination URL.
            doc_id:       RAG-Anything document ID.
            document_id:  Caller-supplied document identifier.
            project_id:   Caller-supplied project identifier.
            status:       "completed" or "failed".
            error:        Human-readable error message when status is "failed".
            max_retries:  Maximum delivery attempts (default 3).

        Returns:
            True if delivered successfully, False if all attempts failed.
        """
        payload = {
            "docId": doc_id,
            "documentId": document_id,
            "projectId": project_id,
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        if error:
            payload["error"] = error

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Sending webhook to {webhook_url} for document {document_id} "
                    f"(attempt {attempt + 1}/{max_retries}, status={status})"
                )

                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    logger.info(f"✅ Webhook delivered for document {document_id}")
                    return True

            except httpx.HTTPError as e:
                wait = 2 ** attempt  # 1s, 2s, 4s
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Webhook attempt {attempt + 1} failed for {document_id}: {e}. "
                        f"Retrying in {wait}s…"
                    )
                    await asyncio.sleep(wait)
                else:
                    logger.error(
                        f"❌ Webhook failed for {document_id} after {max_retries} attempts: {e}"
                    )
                    return False

            except Exception as e:
                logger.error(f"❌ Unexpected webhook error for {document_id}: {e}")
                logger.exception(e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return False

        return False
