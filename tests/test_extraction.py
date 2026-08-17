"""
Integration tests for extraction.py's model calls -- exercised against
real sample images and a real running Ollama instance (not mocked), same
pattern as test_triage.py and test_weight_detection.py. Slower than the
pure-logic tests in test_validation.py, since each assertion triggers at
least one live model call.

Requires: sample_labels/ generated (scripts/generate_test_labels.py) and
Ollama running locally with qwen2.5:14b and qwen2.5vl:7b pulled.
"""

import os

import pytest

from app.extraction import (
    FIELD_NAMES,
    careful_extract,
    check_warning_header,
    fast_extract,
)

SAMPLE_DIR = "sample_labels"


def _skip_if_missing(fname):
    path = os.path.join(SAMPLE_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} not found -- run scripts/generate_test_labels.py first")
    return path


def test_fast_extract_returns_all_expected_keys():
    path = _skip_if_missing("old_tom_bourbon_clean.jpg")
    fields = fast_extract(path)
    for key in FIELD_NAMES:
        assert key in fields, f"missing key: {key}"


def test_fast_extract_brand_name_on_clean_compliant_label():
    """Brand name has been the single most stable field across every test
    run this session -- a good canary for 'is the fast path pipeline
    working at all', not exhaustive field-by-field validation."""
    path = _skip_if_missing("old_tom_bourbon_clean.jpg")
    fields = fast_extract(path)
    assert fields["brand_name"] is not None
    assert "OLD TOM" in fields["brand_name"].upper()


def test_careful_extract_returns_all_expected_keys():
    path = _skip_if_missing("old_tom_bourbon_messy.jpg")
    fields = careful_extract(path)
    for key in FIELD_NAMES:
        assert key in fields, f"missing key: {key}"


def test_check_warning_header_returns_exactly_two_keys():
    path = _skip_if_missing("old_tom_bourbon_clean.jpg")
    result = check_warning_header(path)
    assert set(result.keys()) == {
        "government_warning_header",
        "government_warning_header_bold",
    }


def test_check_warning_header_bold_is_deterministic_type():
    """government_warning_header_bold must always be a real bool or None
    -- never a string or other JSON-ish value that would crash
    validation.py's boolean_required branch."""
    path = _skip_if_missing("old_tom_bourbon_clean.jpg")
    result = check_warning_header(path)
    assert result["government_warning_header_bold"] in (True, False, None)