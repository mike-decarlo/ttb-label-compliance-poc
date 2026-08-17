"""
LLM backend abstraction.

extraction.py's prompts, orchestration, and JSON-parsing logic stay
completely provider-agnostic -- this module is the only place that knows
HOW to actually call a model. Two implementations:

  OllamaBackend  -- local inference (dev default), zero network dependency
  GeminiBackend  -- hosted, free-tier API (for the public Streamlit deployment)

Selected via the LLM_BACKEND environment variable ("ollama" or "gemini"),
defaulting to "ollama" so local dev behavior is unchanged unless
explicitly opted into the hosted path.
"""

import os
from functools import lru_cache
from typing import Protocol

import ollama
from PIL import Image

MODEL_OPTIONS = {"temperature": 0, "seed": 42}


class LLMBackend(Protocol):
    def complete_text(self, prompt: str) -> str:
        """Send a text-only prompt, return the raw response text."""
        ...

    def complete_vision(self, prompt: str, image_path: str) -> str:
        """Send a prompt plus an image, return the raw response text."""
        ...

    def read_document_text(self, image_path: str) -> str:
        """Read raw text from a document image, using whatever strategy
        this backend has available for it."""
        ...


class OllamaBackend:
    def __init__(self, fast_model: str = "qwen2.5:14b", vision_model: str = "qwen2.5vl:7b",
                 ocr_model: str = "glm-ocr"):
        self.fast_model = fast_model
        self.vision_model = vision_model
        self.ocr_model = ocr_model

    def complete_text(self, prompt: str) -> str:
        response = ollama.chat(
            model=self.fast_model,
            messages=[{"role": "user", "content": prompt}],
            format="json",
            options=MODEL_OPTIONS,
        )
        return response["message"]["content"]

    def complete_vision(self, prompt: str, image_path: str) -> str:
        response = ollama.chat(
            model=self.vision_model,
            messages=[{"role": "user", "content": prompt, "images": [image_path]}],
            format="json",
            options=MODEL_OPTIONS,
        )
        return response["message"]["content"]

    def read_document_text(self, image_path: str) -> str:
        """
        Reads raw text using GLM-OCR, a small OCR-specialist model --
        confirmed more reliable than a general vision-model prompt on
        messy/degraded label images (correct core-field extraction
        across all 5 messy test images during development). GLM-OCR has
        been observed emitting its full transcription twice back-to-back
        in a single response; _dedupe_repeated_text handles that.
        """
        response = ollama.chat(
            model=self.ocr_model,
            messages=[{"role": "user", "content": "Extract all text from this document image.",
                       "images": [image_path]}],
            options={"temperature": 0},
        )
        return _dedupe_repeated_text(response["message"]["content"])


def _dedupe_repeated_text(text: str) -> str:
    """GLM-OCR-specific quirk: finds where the first line repeats again
    later in the text and truncates there, rather than depending on a
    token cap tuned to any particular document's length."""
    text = text.strip()
    first_line = text.split("\n")[0]
    if not first_line:
        return text
    second_occurrence = text.find(first_line, len(first_line))
    return text[:second_occurrence].strip() if second_occurrence != -1 else text


class GeminiBackend:
    """
    Hosted backend for the public deployment. One model handles both text
    and vision calls -- unlike the local setup, Gemini's Flash models are
    natively multimodal, so there's no separate "fast model."

    Free-tier rate limits shift over time -- check
    https://ai.google.dev/gemini-api/docs/rate-limits for current numbers
    rather than assuming a fixed throughput. Also worth remembering: the
    concurrency this project uses today (batch.py's ThreadPoolExecutor)
    was tuned for local GPU limits, not a per-minute API cap -- that
    needs its own look before this backend is used for real batches.
    """

    def __init__(self, model: str = "gemini-3.6-flash"):
        # Imported here, not at module level, so local-only dev never
        # needs google-genai installed unless this backend is selected.
        from google import genai
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
        self.model = model

    def complete_text(self, prompt: str) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text

    def complete_vision(self, prompt: str, image_path: str) -> str:
        from google.genai import types
        response = self.client.models.generate_content(
            model=self.model,
            contents=[prompt, Image.open(image_path)],
            config=types.GenerateContentConfig(temperature=0),
        )
        return response.text

    def read_document_text(self, image_path: str) -> str:
        """
        No GLM-OCR equivalent on the hosted path -- falls back to asking
        the vision model to transcribe the document directly. UNTESTED:
        we have real evidence GLM-OCR reads messy images well; we have
        none yet on whether this fallback performs comparably. Treat
        this as a functional default, not a validated one, until it's
        actually been checked.
        """
        return self.complete_vision(
            "Extract all text from this document image, verbatim.", image_path
        )


@lru_cache(maxsize=1)
def get_backend() -> LLMBackend:
    """Selected once per process via LLM_BACKEND; cached so repeated
    calls within a batch reuse the same client rather than reconstructing
    it per label."""
    backend_name = os.environ.get("LLM_BACKEND", "ollama").lower()
    if backend_name == "ollama":
        return OllamaBackend()
    if backend_name == "gemini":
        return GeminiBackend()
    raise ValueError(f"Unknown LLM_BACKEND: {backend_name!r} (expected 'ollama' or 'gemini')")