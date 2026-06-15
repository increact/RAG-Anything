"""Tests for the pure helpers in raganything.document_worker.

Skipped automatically when the raganything package is not importable.
"""
from datetime import datetime, timedelta, timezone

import pytest

document_worker = pytest.importorskip("raganything.document_worker")


# --- _presigned_url_expired ------------------------------------------------

def test_presigned_url_expired_false_for_plain_url():
    assert document_worker._presigned_url_expired("https://s3.example.com/file.pdf") is False


def test_presigned_url_expired_true_for_expired_signature():
    # Signed in the distant past with a 1h validity window.
    url = (
        "https://s3.example.com/file.pdf"
        "?X-Amz-Date=20200101T000000Z&X-Amz-Expires=3600"
    )
    assert document_worker._presigned_url_expired(url) is True


def test_presigned_url_expired_false_for_valid_signature():
    signed = datetime.now(timezone.utc) - timedelta(minutes=1)
    url = (
        "https://s3.example.com/file.pdf"
        f"?X-Amz-Date={signed.strftime('%Y%m%dT%H%M%SZ')}&X-Amz-Expires=3600"
    )
    assert document_worker._presigned_url_expired(url) is False


def test_presigned_url_expired_false_on_malformed_params():
    url = "https://s3.example.com/file.pdf?X-Amz-Date=garbage&X-Amz-Expires=abc"
    assert document_worker._presigned_url_expired(url) is False


# --- _maybe_dockerize_webhook ----------------------------------------------

def test_maybe_dockerize_webhook_noop_when_disabled(monkeypatch):
    monkeypatch.delenv("DOCKER_NETWORKING", raising=False)
    url = "http://localhost:3000/cb"
    assert document_worker._maybe_dockerize_webhook(url) == url


def test_maybe_dockerize_webhook_rewrites_when_enabled(monkeypatch):
    monkeypatch.setenv("DOCKER_NETWORKING", "true")
    out = document_worker._maybe_dockerize_webhook("http://localhost:3000/cb")
    assert out == "http://host.docker.internal:3000/cb"


def test_maybe_dockerize_webhook_leaves_non_localhost(monkeypatch):
    monkeypatch.setenv("DOCKER_NETWORKING", "true")
    url = "https://hooks.example.com/cb"
    assert document_worker._maybe_dockerize_webhook(url) == url


def test_maybe_dockerize_webhook_handles_empty(monkeypatch):
    monkeypatch.setenv("DOCKER_NETWORKING", "true")
    assert document_worker._maybe_dockerize_webhook("") == ""
