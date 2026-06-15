"""Tests for api_models — API request/response schemas."""
import api_models


def test_document_metadata_vlm_model_defaults_none():
    m = api_models.DocumentMetadata(parser="mineru", parse_method="auto", doc_id="d1")
    assert m.vlm_model is None


def test_document_metadata_vlm_model_set():
    m = api_models.DocumentMetadata(parser="mineru", parse_method="auto",
                                    doc_id="d1", vlm_model="openai/gpt-4o-mini")
    assert m.vlm_model == "openai/gpt-4o-mini"


def test_result_metadata_vlm_model_defaults_none():
    m = api_models.ResultMetadata(parser="mineru", parse_method="auto", doc_id="d1")
    assert m.vlm_model is None
