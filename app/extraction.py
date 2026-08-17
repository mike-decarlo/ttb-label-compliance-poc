"""
Field extraction: turns a label image into a dict of extracted field values.

TWO PATHS, chosen by triage.score_image_quality():

  fast_extract()     -- OCR the clean image, then a local LLM (via Ollama)
                         parses the raw text into structured fields.
  careful_extract()  -- skip OCR; a local vision-capable model reads the
                         image directly.

Both run fully locally through Ollama -- no outbound network calls at
inference time, which meaningfully changes the TTB firewall story for
this part of the pipeline (see README).

NOTE ON MODELS: qwen2.5:14b (fast_extract) is text-only -- it cannot read
images. Vision needs a separate model family, qwen2.5vl, which tops out
at 32b (no 14b vision size exists). Defaulting to qwen2.5vl:7b for speed;
swap to qwen2.5vl:32b if accuracy on messy images needs to improve and
the hardware can take it.
"""

import json

import ollama
import pytesseract
from PIL import Image

from app.weight_detection import classify_header_bold

FAST_MODEL = "qwen2.5:14b"
VISION_MODEL = "qwen2.5vl:7b"

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

# Deterministic sampling: repeated runs of the same image should extract
# identical text and produce identical verdicts. temperature=0 removes
# most randomness; seed pins what little remains.
MODEL_OPTIONS = {"temperature": 0, "seed": 42}


def _safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Model didn't return clean JSON -- fail safe rather than crash the batch.
        return {k: None for k in FIELD_NAMES}


def check_warning_header(image_path: str) -> dict:
    """
    Header wording/casing: read by the vision model (one small call).
    Bold-ness: measured deterministically via app.weight_detection, not
    judged by the model at all. Used by both fast_extract() and
    careful_extract() -- one shared implementation of this check.
    """
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": HEADER_TEXT_SCHEMA,
            "images": [image_path],
        }],
        format="json",
        options=MODEL_OPTIONS,
    )
    parsed = _safe_json_parse(response["message"]["content"])
    return {
        "government_warning_header": parsed.get("government_warning_header"),
        "government_warning_header_bold": classify_header_bold(image_path),
    }


def fast_extract(image_path: str) -> dict:
    """
    OCR a clean label, then use a local LLM to parse the raw text into
    fields. Header info comes from check_warning_header() -- the one
    part of an otherwise "fast" submission that still looks at the image.
    """
    raw_text = pytesseract.image_to_string(Image.open(image_path))
    response = ollama.chat(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": f"{FIELD_SCHEMA_CORE}\n\nLabel text:\n{raw_text}"}],
        format="json",
        options=MODEL_OPTIONS,
    )
    fields = _safe_json_parse(response["message"]["content"])
    fields.update(check_warning_header(image_path))
    return fields


def careful_extract(image_path: str) -> dict:
    """
    Messy images skip OCR entirely -- a local vision model reading the
    image directly handles skew/glare/lighting better than OCR-then-parse.
    Header info is delegated to check_warning_header(), same as fast_extract().
    """
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": FIELD_SCHEMA_CORE,
            "images": [image_path],
        }],
        format="json",
        options=MODEL_OPTIONS,
    )
    fields = _safe_json_parse(response["message"]["content"])
    fields.update(check_warning_header(image_path))
    return fields