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
- government_warning: the BODY of the warning statement only -- everything
  from "(1)" onward. Do NOT include the "GOVERNMENT WARNING:" header itself.
  Preserve normal word spacing even where the label's text wraps across
  multiple lines -- never join or split words at a line break.
"""

# The header's wording/casing and its bold-ness are asked separately from
# each other, deliberately:
#   - government_warning_header is TEXT, transcribed verbatim. This lets
#     validation.py do a real, deterministic, case-sensitive string
#     comparison -- all-caps compliance is a property of the extracted
#     characters, not something the model needs to self-judge as true/false.
#   - government_warning_header_bold is the one piece that genuinely can't
#     be reduced to extracted text -- font weight isn't a character -- so
#     it stays a model judgment, but now it's asked on its own, not
#     bundled with wording/casing.
HEADER_QUESTION = """
Also include these two keys:
- government_warning_header: transcribe EXACTLY, character-for-character,
  the heading text that introduces the government warning (the short
  phrase immediately before "(1)..." begins, typically ending in a
  colon). Preserve the exact capitalization as printed -- if it is title
  case, transcribe it in title case; if it is all caps, transcribe it in
  all caps. Do not correct, normalize, or re-case what you transcribe.
- government_warning_header_bold: true if that heading text is visibly
  bolder/heavier weight than the surrounding body text; false if it is
  the same weight as the body text; null only if the image is too
  degraded to judge weight at all.
"""

HEADER_ONLY_SCHEMA = f"""Look at this alcohol label image. Return ONLY a
JSON object with exactly these keys:
{HEADER_QUESTION}"""

# Deterministic sampling: repeated runs of the same image should extract
# identical text and produce identical verdicts. temperature=0 removes
# most randomness; seed pins what little remains.
MODEL_OPTIONS = {"temperature": 0, "seed": 42}


def fast_extract(image_path: str) -> dict:
    """
    OCR a clean label, then use a local LLM to parse the raw text into
    fields. The header's wording/casing and bold-ness can't be judged from
    OCR text alone, so a second, narrow vision call handles those -- the
    one part of an otherwise "fast" submission that still looks at the
    image.
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
    Formatting is answered in this same call, since the model is already
    looking at the image for everything else -- unlike fast_extract(),
    no second call is needed here.
    """
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": FIELD_SCHEMA_CORE + HEADER_QUESTION,
            "images": [image_path],
        }],
        format="json",
        options=MODEL_OPTIONS,
    )
    return _safe_json_parse(response["message"]["content"])


def check_warning_header(image_path: str) -> dict:
    """
    A narrow, two-question vision call: what does the warning header
    literally say (verbatim, exact casing), and is it bold? Used only by
    fast_extract() -- careful_extract() already answers both as part of
    its one full-image call.
    """
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": HEADER_ONLY_SCHEMA,
            "images": [image_path],
        }],
        format="json",
        options=MODEL_OPTIONS,
    )
    parsed = _safe_json_parse(response["message"]["content"])
    return {
        "government_warning_header": parsed.get("government_warning_header"),
        "government_warning_header_bold": parsed.get("government_warning_header_bold"),
    }


def _safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Model didn't return clean JSON -- fail safe rather than crash the batch.
        return {k: None for k in FIELD_NAMES}