"""Tests for DoclingParser._docling_page_idx.

Verifies real page numbers are extracted from a Docling block's `prov`
provenance instead of the previous fabricated `cnt // 10` value.

Skipped automatically when the raganything package is not importable.
"""
import pytest

parser_mod = pytest.importorskip("raganything.parser")
DoclingParser = parser_mod.DoclingParser


def test_page_idx_from_prov_is_zero_based():
    # Docling stores 1-indexed page_no; page_idx is 0-indexed.
    block = {"prov": [{"page_no": 5}]}
    assert DoclingParser._docling_page_idx(block) == 4


def test_page_idx_first_page():
    block = {"prov": [{"page_no": 1}]}
    assert DoclingParser._docling_page_idx(block) == 0


def test_page_idx_defaults_to_zero_without_prov():
    assert DoclingParser._docling_page_idx({}) == 0
    assert DoclingParser._docling_page_idx({"prov": []}) == 0


def test_page_idx_defaults_to_zero_on_missing_page_no():
    assert DoclingParser._docling_page_idx({"prov": [{"bbox": [0, 0, 1, 1]}]}) == 0


def test_page_idx_never_negative():
    # Defensive: a malformed page_no=0 must not yield -1.
    assert DoclingParser._docling_page_idx({"prov": [{"page_no": 0}]}) == 0
