"""Unit tests for validation.py — the security-critical input validators.

These cover the path-traversal and SSRF fixes; they import only `validation`
(stdlib + fastapi) so they run without the heavy raganything/ML dependencies.
"""
import pytest
from fastapi import HTTPException

from validation import (
    validate_doc_id,
    validate_external_url,
    redact_url,
    safe_tempfile_suffix,
)


# --- validate_doc_id -------------------------------------------------------

@pytest.mark.parametrize("doc_id", [
    "abc123",
    "doc-1e81dc52a3c31d2f5c35feff2e28e025",
    "A_B-c_9",
    "x" * 128,
])
def test_validate_doc_id_accepts_safe_ids(doc_id):
    assert validate_doc_id(doc_id) == doc_id


@pytest.mark.parametrize("doc_id", [
    "../../etc/passwd",      # path traversal
    "../secret",
    "a/b",                   # slash
    "a\\b",                  # backslash
    "with space",
    "semi;colon",
    "",                      # empty
    "x" * 129,               # too long
    "unicodeé",
])
def test_validate_doc_id_rejects_unsafe_ids(doc_id):
    with pytest.raises(HTTPException) as exc:
        validate_doc_id(doc_id)
    assert exc.value.status_code == 400


# --- validate_external_url -------------------------------------------------

def test_validate_external_url_rejects_empty():
    with pytest.raises(HTTPException) as exc:
        validate_external_url("", "webhook_url")
    assert exc.value.status_code == 400


@pytest.mark.parametrize("url", [
    "ftp://example.com/x",
    "file:///etc/passwd",
    "gopher://example.com",
    "not-a-url",
])
def test_validate_external_url_rejects_bad_schemes(url):
    with pytest.raises(HTTPException) as exc:
        validate_external_url(url, "webhook_url")
    assert exc.value.status_code == 400


@pytest.mark.parametrize("host", ["metadata.google.internal", "metadata"])
def test_validate_external_url_rejects_metadata_hosts(host):
    with pytest.raises(HTTPException):
        validate_external_url(f"http://{host}/computeMetadata/v1/", "webhook_url")


def test_validate_external_url_allows_public_https():
    # Should not raise.
    validate_external_url("https://api.example.com/webhooks/cb", "webhook_url")


def test_validate_external_url_allows_localhost_in_dev(monkeypatch):
    monkeypatch.delenv("PUBLIC_URL_ONLY", raising=False)
    # Dev default: localhost callbacks are permitted.
    validate_external_url("http://localhost:3000/cb", "webhook_url")


def test_validate_external_url_blocks_private_ip_when_public_only(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL_ONLY", "true")
    # AWS IMDS endpoint — the SSRF target the fix is meant to stop.
    with pytest.raises(HTTPException) as exc:
        validate_external_url("https://169.254.169.254/latest/meta-data/", "s3_url")
    assert exc.value.status_code == 400


def test_validate_external_url_blocks_http_when_public_only(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL_ONLY", "true")
    with pytest.raises(HTTPException):
        validate_external_url("http://api.example.com/cb", "webhook_url")


def test_validate_external_url_blocks_loopback_when_public_only(monkeypatch):
    monkeypatch.setenv("PUBLIC_URL_ONLY", "true")
    with pytest.raises(HTTPException):
        validate_external_url("https://127.0.0.1/cb", "webhook_url")


# --- redact_url ------------------------------------------------------------

def test_redact_url_strips_query_and_fragment():
    redacted = redact_url("https://hook.example.com/cb?token=secret123#frag")
    assert "secret123" not in redacted
    assert redacted == "https://hook.example.com/cb"


def test_redact_url_handles_empty_and_none():
    assert redact_url("") == ""
    assert redact_url(None) == ""


# --- safe_tempfile_suffix --------------------------------------------------

@pytest.mark.parametrize("filename,expected", [
    ("report.pdf", ".pdf"),
    ("IMAGE.PNG", ".png"),
    ("doc.docx", ".docx"),
    ("noext", ""),
    ("", ""),
    (None, ""),
])
def test_safe_tempfile_suffix_allowed(filename, expected):
    assert safe_tempfile_suffix(filename) == expected


@pytest.mark.parametrize("filename", [
    "payload.sh",
    "evil.exe",
    "x.bat",
    "../../etc/cron.d/payload.sh",
])
def test_safe_tempfile_suffix_rejects_disallowed(filename):
    with pytest.raises(HTTPException) as exc:
        safe_tempfile_suffix(filename)
    assert exc.value.status_code == 400
