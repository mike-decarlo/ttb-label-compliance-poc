"""
Tests for app/llm_backend.py -- backend SELECTION logic only. Does not
test actual model calls (those are already covered by
tests/test_extraction.py against a real running Ollama instance) -- this
only verifies the right backend class gets picked, without needing
network access or API keys.
"""

import pytest

from app.llm_backend import OllamaBackend, get_backend


@pytest.fixture(autouse=True)
def _clear_backend_cache_and_env(monkeypatch):
    """get_backend() is cached via lru_cache -- clear it before AND after
    each test so one test's cached backend can't leak into the next."""
    get_backend.cache_clear()
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    yield
    get_backend.cache_clear()


def test_defaults_to_ollama_when_unset():
    assert isinstance(get_backend(), OllamaBackend)


def test_selects_ollama_explicitly(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    assert isinstance(get_backend(), OllamaBackend)


def test_selection_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "OLLAMA")
    assert isinstance(get_backend(), OllamaBackend)


def test_unknown_backend_raises_clear_error(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "not_a_real_backend")
    with pytest.raises(ValueError, match="Unknown LLM_BACKEND"):
        get_backend()


def test_gemini_backend_requires_api_key(monkeypatch):
    """Constructing GeminiBackend without GEMINI_API_KEY set should fail
    clearly rather than silently proceeding with no credentials."""
    monkeypatch.setenv("LLM_BACKEND", "gemini")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        get_backend()


def test_ollama_backend_has_read_document_text_method():
    assert hasattr(OllamaBackend(), "read_document_text")


def test_dedupe_repeated_text_removes_second_copy():
    from app.llm_backend import _dedupe_repeated_text
    doubled = "BRAND NAME\nSome text here.\nBRAND NAME\nSome text here."
    assert _dedupe_repeated_text(doubled) == "BRAND NAME\nSome text here."


def test_dedupe_repeated_text_leaves_single_copy_unchanged():
    from app.llm_backend import _dedupe_repeated_text
    single = "BRAND NAME\nSome text here."
    assert _dedupe_repeated_text(single) == single


def test_gemini_backend_has_read_document_text_method(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-attribute-check")
    from app.llm_backend import GeminiBackend
    assert hasattr(GeminiBackend(), "read_document_text")