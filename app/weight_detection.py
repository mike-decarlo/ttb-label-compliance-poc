"""
Deterministic bold/regular classification for the government warning
header.

Replaces a vision-model judgment that consistently failed to discriminate
bold from regular text -- confirmed even on a sharp, undegraded image
where the difference was visually unambiguous. Bold-ness is a measurable
property (ink density), not just a perceptual one, so this uses OCR to
locate the header line and OpenCV to measure it, calibrated against two
freshly-rendered reference images rather than a fixed magic threshold.
"""

import io
import re
from functools import lru_cache

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw
from pytesseract import Output
from rapidfuzz import fuzz

from app.deskew import deskew
from app.fonts import load_font


@lru_cache(maxsize=8)
def reference_weight_images(font_size: int) -> tuple[bytes, bytes]:
    """
    Renders two reference images -- one bold, one regular -- AT THE SAME
    FONT SIZE as the header being measured, cropped TIGHTLY to the
    rendered text's own bounding box.

    Both the size match and the crop tightness matter: ink density as a
    ratio isn't scale-invariant, AND it isn't padding-invariant. OCR
    hands us a header crop with almost no background margin, so
    comparing it against a reference measured over a mostly-blank canvas
    would report a huge density gap that reflects padding, not font
    weight -- which is exactly what was happening before this fix.
    Cached per size, so repeated calls at the same size don't re-render.
    """
    def render(bold: bool) -> bytes:
        font = load_font(font_size, bold=bold)
        text = "GOVERNMENT WARNING:"
        scratch = Image.new("RGB", (600, font_size * 3), "white")
        draw = ImageDraw.Draw(scratch)
        bbox = draw.textbbox((10, 10), text, font=font)
        draw.text((10, 10), text, fill="black", font=font)
        cropped = scratch.crop(bbox)
        buf = io.BytesIO()
        cropped.save(buf, format="PNG")
        return buf.getvalue()

    return render(bold=True), render(bold=False)


def _ink_density(image: Image.Image, bbox: tuple[int, int, int, int] | None = None) -> float:
    """
    Ratio of dark (ink) pixels to total pixels in the region, after Otsu
    thresholding. Bold text has measurably higher ink density than
    regular text at the same size/character count.
    """
    gray = image.convert("L")
    if bbox is not None:
        left, top, w, h = bbox
        gray = gray.crop((left, top, left + w, top + h))
    arr = np.array(gray)
    if arr.size == 0:
        return 0.0
    _, binary = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.count_nonzero(binary)) / binary.size


def _find_header_words(data: dict, threshold: float = 70.0) -> tuple[int, ...] | None:
    """
    Finds OCR word indices matching "GOVERNMENT" and/or "WARNING" via
    fuzzy comparison. Returns indices for whichever word(s) clear the
    threshold -- both if both are found (most precise), just one if only
    one is. This is deliberately an OR, not an AND: real diagnostic data
    showed "WARNING" reliably detected while "GOVERNMENT" was never
    detected at all across every test image, so requiring both was the
    actual bottleneck, not a general blur problem. A crop of "WARNING:"
    alone is still representative of the header's font weight, since
    both words render at the same weight. Returns None only if neither
    word is found.
    """
    best_gov, best_gov_score = None, threshold
    best_warn, best_warn_score = None, threshold
    for i, raw_word in enumerate(data["text"]):
        word = re.sub(r"[^A-Za-z]", "", raw_word).upper()
        if not word:
            continue
        gov_score = fuzz.ratio(word, "GOVERNMENT")
        if gov_score > best_gov_score:
            best_gov, best_gov_score = i, gov_score
        warn_score = fuzz.ratio(word, "WARNING")
        if warn_score > best_warn_score:
            best_warn, best_warn_score = i, warn_score

    found = tuple(i for i in (best_gov, best_warn) if i is not None)
    return found if found else None


def _bbox_from_word_indices(data: dict, indices: tuple[int, ...]) -> tuple[int, int, int, int]:
    lefts = [data["left"][i] for i in indices]
    tops = [data["top"][i] for i in indices]
    rights = [data["left"][i] + data["width"][i] for i in indices]
    bottoms = [data["top"][i] + data["height"][i] for i in indices]
    return (min(lefts), min(tops), max(rights) - min(lefts), max(bottoms) - min(tops))


def locate_header_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Finds the pixel bounding box of the "GOVERNMENT WARNING" header via
    fuzzy OCR word matching on the raw image. No preprocessing fallback --
    two independent attempts (aggressive sharpening, then 2x upscale with
    mild CLAHE) were tested and both measurably found FEWER usable words
    than the raw pass on the same images, not more. Enlarging or
    sharpening an already-blurred image appears to amplify the existing
    blur rather than recover real detail. Returns None -- fail closed --
    if neither "GOVERNMENT" nor "WARNING" can be found even fuzzily.
    """
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    match = _find_header_words(data)
    return _bbox_from_word_indices(data, match) if match else None


def classify_header_bold(image_path: str) -> bool | None:
    """
    True/False if the header's measured ink density is closer to the
    bold or regular reference; None if the header line couldn't be
    located at all -- fails closed (flagged for review), not guessed.

    DESIGN PRINCIPLE, not just an implementation detail: a false "not
    bold" (correctly-formatted label sent to manual review) is an
    acceptable cost. A false "bold" (non-compliant label silently
    approved) is not, ever. Three bounding-box refinement strategies
    were tried and rejected specifically because each one traded a
    safe-direction fix for a dangerous-direction regression on at least
    one test case -- see git history / README for details. Any future
    change to this function must be evaluated against that asymmetry,
    not against raw accuracy alone.

    The image is deskewed once, up front -- messy images can be rotated
    several degrees, which fragments OCR word detection independent of
    blur. The SAME deskewed copy is used for both locating the header
    and measuring its density; locating on one version and measuring on
    another would silently crop the wrong region.
    """
    image = deskew(Image.open(image_path))
    bbox = locate_header_bbox(image)
    if bbox is None:
        return None

    header_density = _ink_density(image, bbox)
    font_size = max(10, bbox[3])  # bbox height as a proxy for the header's font size
    bold_bytes, regular_bytes = reference_weight_images(font_size)
    bold_density = _ink_density(Image.open(io.BytesIO(bold_bytes)))
    regular_density = _ink_density(Image.open(io.BytesIO(regular_bytes)))

    midpoint = (bold_density + regular_density) / 2
    return header_density >= midpoint