"""
Input-validation helpers for the RAG-Anything API server.

Kept in a standalone module (no heavy imports) so the security-critical
validators can be unit-tested without importing the whole FastAPI app or the
raganything package.
"""
import os
import re
import ipaddress
import urllib.parse
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

# Client-supplied IDs are used as filesystem path components and as keys in
# Redis; restrict to a safe character set to prevent path traversal.
_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")

# Suffixes that may be passed to tempfile.NamedTemporaryFile. Driven from what
# the parsers actually accept; an unknown suffix coming from a client filename
# should not be propagated to disk.
ALLOWED_FILE_SUFFIXES = frozenset({
    ".pdf", ".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx",
    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif", ".webp",
    ".txt", ".md", ".html", ".htm", ".csv", ".rtf", ".odt",
})


def _public_url_only() -> bool:
    """Whether to enforce the public-only URL policy on webhook/S3 URLs.

    Read per-call (not cached at import) so it is easy to exercise both
    branches in tests via monkeypatch.setenv. Off in local dev (where
    http://localhost:3000/... is normal), should be true in production.
    """
    return os.getenv("PUBLIC_URL_ONLY", "false").lower() in ("true", "1", "yes")


def validate_external_url(url: str, field: str) -> None:
    """Reject obviously dangerous URLs supplied by clients.

    Blocks: non-http(s) schemes, plain hostnames matching common
    metadata-service names, and — when PUBLIC_URL_ONLY is set —
    private/loopback/link-local IPs and non-https schemes.
    """
    if not url:
        raise HTTPException(400, f"{field} is required")
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(400, f"{field} must use http(s)")
    public_only = _public_url_only()
    if public_only and parsed.scheme != "https":
        raise HTTPException(400, f"{field} must use https")
    host = (parsed.hostname or "").lower()
    if not host:
        raise HTTPException(400, f"{field} has no host")
    if host in {"metadata.google.internal", "metadata"}:
        raise HTTPException(400, f"{field} resolves to a metadata service")
    try:
        addr = ipaddress.ip_address(host)
        if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_multicast:
            if public_only:
                raise HTTPException(400, f"{field} resolves to a non-public address")
    except ValueError:
        # Not a literal IP; full DNS-rebinding protection is out of scope here.
        pass


def validate_doc_id(doc_id: str) -> str:
    """Validate a client-supplied identifier used as a path component."""
    if not _SAFE_ID_RE.fullmatch(doc_id or ""):
        raise HTTPException(400, "Invalid doc_id: must be 1-128 chars of [A-Za-z0-9_-]")
    return doc_id


def redact_url(url: Optional[str]) -> str:
    """Return scheme://host/path of `url` with query/fragment removed so
    secrets in query parameters do not land in logs."""
    if not url:
        return ""
    try:
        p = urllib.parse.urlparse(url)
        return f"{p.scheme}://{p.netloc}{p.path}"
    except Exception:
        return "<unparseable url>"


def safe_tempfile_suffix(filename: Optional[str]) -> str:
    """Return an allowlisted suffix for tempfile.NamedTemporaryFile."""
    if not filename:
        return ""
    suffix = Path(filename).suffix.lower()
    if suffix and suffix not in ALLOWED_FILE_SUFFIXES:
        raise HTTPException(400, f"Unsupported file type: {suffix}")
    return suffix
