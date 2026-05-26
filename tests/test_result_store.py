"""Tests for raganything.result_store — the on-disk result persistence layer.

Skipped automatically when the raganything package (and its heavy ML deps)
is not importable, e.g. in a minimal CI environment.
"""
import pytest

result_store = pytest.importorskip("raganything.result_store")


def _save(tmp_path, doc_id="doc-abc", **overrides):
    kwargs = dict(
        output_dir=str(tmp_path),
        doc_id=doc_id,
        document_id="client-doc-1",
        project_id="proj-1",
        markdown="# Title\n\nbody",
        content_list=[{"type": "text", "text": "body"}],
        metadata={"parser": "mineru"},
    )
    kwargs.update(overrides)
    result_store.save_result(**kwargs)
    return kwargs


def test_save_then_load_roundtrip(tmp_path):
    _save(tmp_path)
    record = result_store.load_result(str(tmp_path), "doc-abc")
    assert record is not None
    assert record["doc_id"] == "doc-abc"
    assert record["document_id"] == "client-doc-1"
    assert record["markdown"] == "# Title\n\nbody"
    assert record["content_list"] == [{"type": "text", "text": "body"}]


def test_load_missing_returns_none(tmp_path):
    assert result_store.load_result(str(tmp_path), "no-such-doc") is None


def test_save_counts_content_types(tmp_path):
    _save(
        tmp_path,
        content_list=[
            {"type": "text", "text": "t"},
            {"type": "table"},
            {"type": "image"},
            {"type": "image"},
            {"type": "equation"},
        ],
        metadata={"parser": "mineru"},
    )
    record = result_store.load_result(str(tmp_path), "doc-abc")
    assert record["metadata"]["tables"] == 1
    assert record["metadata"]["images"] == 2
    assert record["metadata"]["formulas"] == 1


def test_save_raises_on_unwritable_path(tmp_path):
    # A path whose parent is a file (not a dir) cannot be created — save_result
    # must raise rather than silently swallow, so the caller does not send a
    # "completed" webhook for a result that was never written.
    blocker = tmp_path / "afile"
    blocker.write_text("x")
    with pytest.raises(Exception):
        result_store.save_result(
            output_dir=str(blocker / "subdir"),
            doc_id="doc-x",
            document_id="d",
            project_id="p",
            markdown="m",
            content_list=[],
            metadata={},
        )
