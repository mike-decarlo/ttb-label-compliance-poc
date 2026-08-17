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


def test_fast_extract_preserves_numbered_warning_markers():
    """Regression guard: FIELD_SCHEMA_CORE must instruct the model to
    include '(1)' and '(2)' in government_warning. A prior version of
    this prompt omitted that instruction, causing the markers to be
    silently dropped from extracted text -- caught via redbridge's
    output, not by any test at the time."""
    path = _skip_if_missing("redbridge_rum_abv_mismatch_clean.jpg")
    fields = fast_extract(path)
    warning = fields.get("government_warning") or ""
    assert "(1)" in warning
    assert "(2)" in warning


def test_safe_json_parse_strips_markdown_fences():
    """Regression guard: Gemini's vision calls have been observed
    wrapping JSON output in ```json ... ``` even when explicitly asked
    for ONLY a JSON object -- confirmed live, not hypothetical. Without
    stripping, this fallback (all fields None) would silently blank out
    every field on every submission."""
    from app.extraction import _safe_json_parse
    fenced = '```json\n{"brand_name": "TEST BRAND"}\n```'
    result = _safe_json_parse(fenced)
    assert result.get("brand_name") == "TEST BRAND"


def test_safe_json_parse_handles_clean_json_unchanged():
    from app.extraction import _safe_json_parse
    clean = '{"brand_name": "TEST BRAND"}'
    result = _safe_json_parse(clean)
    assert result.get("brand_name") == "TEST BRAND"


def test_careful_extract_brand_name_on_messy_compliant_label():
    """Now that GLM-OCR handles messy-image text reading, this should
    resolve correctly rather than being routed to the vision model's
    direct read -- confirmed across all 5 messy test images during
    development."""
    path = _skip_if_missing("old_tom_bourbon_messy.jpg")
    fields = careful_extract(path)
    assert fields["brand_name"] is not None
    assert "OLD TOM" in fields["brand_name"].upper()


def test_careful_extract_delegates_to_backend_read_document_text(monkeypatch):
    """Default (OCR_ENGINE unset) should call get_backend().read_document_text()
    and nothing else -- extraction.py shouldn't know which engine that
    resolves to."""
    import app.extraction as extraction_module
    monkeypatch.delenv("OCR_ENGINE", raising=False)

    class FakeBackend:
        def read_document_text(self, image_path):
            return "FAKE RAW TEXT"
        def complete_text(self, prompt):
            assert "FAKE RAW TEXT" in prompt
            return '{"brand_name": "FROM BACKEND"}'
        def complete_vision(self, prompt, image_path):
            return "{}"

    monkeypatch.setattr(extraction_module, "get_backend", lambda: FakeBackend())
    result = extraction_module.careful_extract("sample_labels/old_tom_bourbon_messy.jpg")
    assert result.get("brand_name") == "FROM BACKEND"


def test_careful_extract_ocr_engine_glmocr_forces_ollama_regardless_of_backend(monkeypatch):
    """OCR_ENGINE=glmocr should force GLM-OCR reading even when the
    active backend (for parsing) is something else entirely."""
    import app.extraction as extraction_module
    monkeypatch.setenv("OCR_ENGINE", "glmocr")

    class FakeOllamaBackend:
        def read_document_text(self, image_path):
            return "FORCED GLMOCR TEXT"

    class FakeActiveBackend:
        def complete_text(self, prompt):
            assert "FORCED GLMOCR TEXT" in prompt
            return '{"brand_name": "FROM FORCED PATH"}'
        def complete_vision(self, prompt, image_path):
            return "{}"
        def read_document_text(self, image_path):
            raise AssertionError("must not be called when OCR_ENGINE=glmocr overrides it")

    monkeypatch.setattr(extraction_module, "OllamaBackend", lambda: FakeOllamaBackend())
    monkeypatch.setattr(extraction_module, "get_backend", lambda: FakeActiveBackend())

    result = extraction_module.careful_extract("sample_labels/old_tom_bourbon_messy.jpg")
    assert result.get("brand_name") == "FROM FORCED PATH"