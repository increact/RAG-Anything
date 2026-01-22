"""
Webhook service with retry mechanism for sending callbacks to client
"""
import asyncio
import httpx
from typing import Optional, Dict, Any, List
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
        markdown: Optional[str] = None,
        content_list: Optional[List[Dict[str, Any]]] = None,
        max_retries: int = 3,
    ) -> bool:
        """
        Send webhook callback (with retry mechanism)
        
        Args:
            webhook_url: Webhook URL to send to
            doc_id: RAG Anything document ID
            document_id: Client document ID
            project_id: Client project ID
            status: Processing status ('completed' or 'failed')
            error: Error message if status is 'failed'
            markdown: Markdown content if status is 'completed' (optional)
            content_list: Content list if status is 'completed' (optional)
            max_retries: Maximum number of retry attempts (default: 3)
        
        Returns:
            bool: True if successful, False if all retries failed
        """
        payload = {
            "docId": doc_id,
            "documentId": document_id,
            "projectId": project_id,
            "status": status,
        }
        
        if error:
            payload["error"] = error
        
        if status == "completed" and (markdown or content_list):
            payload["metadata"] = {}
            if markdown:
                payload["metadata"]["markdown"] = markdown
            if content_list:
                payload["metadata"]["contentList"] = content_list
        
        for attempt in range(max_retries):
            try:
                logger.info(
                    f"Sending webhook to {webhook_url} for document {document_id} "
                    f"(attempt {attempt + 1}/{max_retries})"
                )
                
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        webhook_url,
                        json=payload,
                        headers={"Content-Type": "application/json"},
                    )
                    response.raise_for_status()
                    logger.info(f"✅ Webhook sent successfully for document {document_id}")
                    return True  # Success
                
            except httpx.HTTPError as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(
                        f"Webhook attempt {attempt + 1} failed for document {document_id}: {str(e)}. "
                        f"Retrying in {wait_time}s..."
                    )
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(
                        f"❌ Failed to send webhook for document {document_id} after {max_retries} attempts: {str(e)}"
                    )
                    return False
            except Exception as e:
                logger.error(
                    f"❌ Unexpected error sending webhook for document {document_id}: {str(e)}"
                )
                logger.exception(e)
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    return False
        
        return False

