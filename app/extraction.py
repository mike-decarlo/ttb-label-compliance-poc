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

import pytesseract
from PIL import Image

from app.llm_backend import get_backend
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
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Model didn't return clean JSON -- fail safe rather than crash the batch.
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
    Messy images skip OCR entirely -- the active backend's vision call
    reads the image directly for the core fields. Header info is
    delegated to check_warning_header(), same as fast_extract().
    """
    response_text = get_backend().complete_vision(FIELD_SCHEMA_CORE, image_path)
    fields = _safe_json_parse(response_text)
    fields.update(check_warning_header(image_path))
    return fields