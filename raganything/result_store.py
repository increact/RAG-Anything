"""
Result store — persists processing results to disk so they can be retrieved
by doc_id after the webhook notification is delivered.

Each completed document is stored as a single JSON file:
    {output_dir}/{doc_id}_result.json

This keeps the implementation simple and dependency-free while allowing the
API endpoint GET /api/v1/result/{doc_id} to serve the full content on demand.
"""
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _result_path(output_dir: str, doc_id: str) -> Path:
    return Path(output_dir) / f"{doc_id}_result.json"


def save_result(
    output_dir: str,
    doc_id: str,
    document_id: str,
    project_id: str,
    markdown: str,
    content_list: List[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> None:
    """
    Persist a completed processing result to disk.

    Args:
        output_dir:   Directory where result files are written.
        doc_id:       RAG-Anything document ID (content hash).
        document_id:  Caller-supplied document identifier.
        project_id:   Caller-supplied project identifier.
        markdown:     Generated markdown string.
        content_list: Structured content blocks returned by the parser.
        metadata:     Processing metadata (parser, tables, formulas, images…).
    """
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        # Count content types if not already computed
        if not any(k in metadata for k in ("tables", "formulas", "images")):
            for item in content_list:
                if isinstance(item, dict):
                    t = item.get("type", "")
                    if t == "table":
                        metadata["tables"] = metadata.get("tables", 0) + 1
                    elif t == "equation":
                        metadata["formulas"] = metadata.get("formulas", 0) + 1
                    elif t == "image":
                        metadata["images"] = metadata.get("images", 0) + 1

        record = {
            "doc_id": doc_id,
            "document_id": document_id,
            "project_id": project_id,
            "processed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "markdown": markdown,
            "content_list": content_list,
        }

        path = _result_path(output_dir, doc_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False)

        logger.info(f"Result saved: {path} (markdown={len(markdown)} chars, blocks={len(content_list)})")
    except Exception as e:
        logger.error(f"Failed to save result for doc_id={doc_id}: {e}")


def load_result(output_dir: str, doc_id: str) -> Optional[Dict[str, Any]]:
    """
    Load a previously saved processing result.

    Returns:
        The result dict, or None if not found.
    """
    path = _result_path(output_dir, doc_id)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load result for doc_id={doc_id}: {e}")
        return None
