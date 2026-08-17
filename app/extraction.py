"""
Field extraction: turns a label image into a dict of extracted field values.

TWO PATHS, chosen by triage.score_image_quality():

  fast_extract()     -- OCR the clean image, then an LLM parses the raw
                         text into structured fields. Header wording is
                         read by a second, narrow vision call; header
                         bold-ness is measured deterministically (see
                         app.weight_detection), not judged by any model.
  careful_extract()  -- skip OCR; a vision-capable model reads the image
                         directly for the core fields, then the same
                         shared header check as fast_extract().

Model calls go through app.llm_backend.get_backend() -- either a local
Ollama backend (dev default) or a hosted API backend (for the public
deployment), selected via the LLM_BACKEND environment variable. This
module's prompts, orchestration, and parsing logic don't change based on
which backend is active.
"""

import json
import os
import re

import pytesseract
from PIL import Image

from app.llm_backend import OllamaBackend, get_backend
from app.weight_detection import classify_header_bold

FIELD_NAMES = (
    "brand_name", "class_type", "alcohol_content", "net_contents",
    "bottler_name_addr", "country_of_origin", "government_warning",
    "government_warning_header", "government_warning_header_bold",
)

FIELD_SCHEMA_CORE = """Extract these fields from the alcohol label. Return ONLY a
JSON object with these exact keys, using null for anything not present:

- brand_name
- class_type
- alcohol_content
- net_contents
- bottler_name_addr
- country_of_origin
- government_warning: the BODY of the warning statement, transcribed
  verbatim starting from the header. Include the numbered markers "(1)"
  and "(2)" exactly as they appear in the source text -- they are part
  of the required wording, not formatting to strip or a boundary to skip
  past. Do NOT include the "GOVERNMENT WARNING:" header phrase itself.
  Preserve normal word spacing even where the label's text wraps across
  multiple lines -- never join or split words at a line break.
"""

# Header wording/casing is read by the vision model -- reliably accurate
# across every test run so far. Bold-ness is NOT asked here at all
# anymore: it's measured deterministically in app.weight_detection,
# since the model consistently failed to discriminate bold from regular
# text, even on a sharp, undegraded image with an unambiguous difference.
HEADER_TEXT_SCHEMA = """Look at this alcohol label image. Return ONLY a
JSON object with this exact key:
- government_warning_header: transcribe EXACTLY, character-for-character,
  the heading text that introduces the government warning (the short
  phrase immediately before "(1)..." begins, typically ending in a
  colon). Preserve the exact capitalization as printed -- if it is title
  case, transcribe it in title case; if it is all caps, transcribe it in
  all caps. Do not correct, normalize, or re-case what you transcribe.
"""


def _safe_json_parse(text: str) -> dict:
    """
    Parses a model's JSON response. Strips a leading/trailing markdown
    code fence first -- confirmed live: Gemini's vision calls have wrapped
    output in ```json ... ``` even when the prompt explicitly asks for
    ONLY a JSON object, while its own text calls (and Ollama's
    format="json") have not shown this. Rather than assume it's specific
    to one backend or call type, strip unconditionally as a safe
    normalization step -- a no-op on text that's already clean JSON.
    """
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Still not valid JSON after stripping -- fail safe rather than
        # crash the batch.
        return {k: None for k in FIELD_NAMES}


def check_warning_header(image_path: str) -> dict:
    """
    Header wording/casing: read by the active backend's vision call.
    Bold-ness: measured deterministically via app.weight_detection, not
    judged by any model. Used by both fast_extract() and careful_extract()
    -- one shared implementation of this check.
    """
    response_text = get_backend().complete_vision(HEADER_TEXT_SCHEMA, image_path)
    parsed = _safe_json_parse(response_text)
    return {
        "government_warning_header": parsed.get("government_warning_header"),
        "government_warning_header_bold": classify_header_bold(image_path),
    }


def fast_extract(image_path: str) -> dict:
    """
    OCR a clean label, then use the active backend to parse the raw text
    into fields. Header info comes from check_warning_header() -- the
    one part of an otherwise "fast" submission that still looks at the
    image.
    """
    raw_text = pytesseract.image_to_string(Image.open(image_path))
    response_text = get_backend().complete_text(
        f"{FIELD_SCHEMA_CORE}\n\nLabel text:\n{raw_text}"
    )
    fields = _safe_json_parse(response_text)
    fields.update(check_warning_header(image_path))
    return fields


def careful_extract(image_path: str) -> dict:
    """
    Reads raw text from the image, then parses it the same way
    fast_extract() does -- differing only in HOW the text is read.
    Normally delegates entirely to the active backend's
    read_document_text() (GLM-OCR on Ollama, confirmed reliable on messy
    images; a direct vision-read fallback on a hosted backend with no
    GLM-OCR equivalent -- see llm_backend.py). This function doesn't
    need to know which backend is active or how it reads text.

    Set OCR_ENGINE=glmocr to force GLM-OCR regardless of which backend
    is handling field-parsing -- e.g. GLM-OCR reading + a hosted backend
    parsing, to test the reading step independently of the parsing model.

    Header info is delegated to check_warning_header(), unrelated to and
    unaffected by any of this -- header bold-ness is measured
    deterministically, not by any model (see weight_detection.py); this
    was tested separately with GLM-OCR too and found not to work.
    """
    if os.environ.get("OCR_ENGINE", "auto").lower() == "glmocr":
        raw_text = OllamaBackend().read_document_text(image_path)
    else:
        raw_text = get_backend().read_document_text(image_path)

    response_text = get_backend().complete_text(f"{FIELD_SCHEMA_CORE}\n\nLabel text:\n{raw_text}")
    fields = _safe_json_parse(response_text)
    fields.update(check_warning_header(image_path))
    return fields