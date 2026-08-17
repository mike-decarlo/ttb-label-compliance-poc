"""
Generates synthetic test labels for the TTB label compliance POC.

Each profile below renders as a matched pair:
  - "_clean"  -- sharp, well-lit, straight -- should route to the fast path
  - "_messy"  -- blurred, low-contrast, skewed, with simulated glare --
                 should route to the careful path

The profiles aren't random -- each one deliberately exercises a specific
validation path already built into app/validation.py:

  old_tom_bourbon          -- every field matches exactly (baseline pass)
  stones_throw_gin         -- brand name case differs -- fuzzy match PASS
  harborview_import_whisky -- import product -- country of origin required
  crestline_vodka_bad_warning -- warning statement in title case, not
                                  all-caps/bold -- exact match must FAIL
                                  (Jenny's interview example)
  redbridge_rum_abv_mismatch  -- stated ABV outside 0.3-point tolerance --
                                  must FAIL

Also writes a deliberately corrupt file, to exercise the "image could
not be opened" error-handling path in app/batch.py.

Run:
    python scripts/generate_test_labels.py
    python main.py --labels sample_labels/ --applications sample_labels/applications.json
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from app.fonts import load_font as _load_font

OUTPUT_DIR = "sample_labels"
IMAGE_SIZE = (900, 1200)  # portrait, roughly bottle-label proportioned

DEFAULT_WARNING = (
    "GOVERNMENT WARNING: (1) According to the Surgeon General, women should "
    "not drink alcoholic beverages during pregnancy because of the risk of "
    "birth defects. (2) Consumption of alcoholic beverages impairs your "
    "ability to drive a car or operate machinery, and may cause health problems."
)

PROFILES = [
    {
        "name": "old_tom_bourbon",
        "description": "Every field matches exactly -- baseline pass.",
        "actual": {
            "brand_name": "OLD TOM DISTILLERY",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45% Alc./Vol. (90 Proof)",
            "net_contents": "750 mL",
            "bottler_name_addr": "Old Tom Distillery, Bardstown, KY",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING,
        },
        "expected": {
            "brand_name": "OLD TOM DISTILLERY",
            "class_type": "Kentucky Straight Bourbon Whiskey",
            "alcohol_content": "45%",
            "net_contents": "750 mL",
            "bottler_name_addr": "Old Tom Distillery, Bardstown, KY",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING,
        },
        "context": {"is_import": False, "abv_required": True},
        "warning_style": "compliant",
    },
    {
        "name": "stones_throw_gin",
        "description": "Brand name case/formatting differs -- fuzzy match should PASS.",
        "actual": {
            "brand_name": "STONE'S THROW",
            "class_type": "London Dry Gin",
            "alcohol_content": "40% Alc./Vol.",
            "net_contents": "750 mL",
            "bottler_name_addr": "Stone's Throw Distilling Co., Portland, OR",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING,
        },
        "expected": {
            "brand_name": "Stone's Throw",
            "class_type": "London Dry Gin",
            "alcohol_content": "40%",
            "net_contents": "750 mL",
            "bottler_name_addr": "Stone's Throw Distilling Co., Portland, OR",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING,
        },
        "context": {"is_import": False, "abv_required": True},
        "warning_style": "compliant",
    },
    {
        "name": "harborview_import_whisky",
        "description": "Imported product -- country of origin is required and must match.",
        "actual": {
            "brand_name": "HARBORVIEW RESERVE",
            "class_type": "Blended Scotch Whisky",
            "alcohol_content": "43% Alc./Vol.",
            "net_contents": "750 mL",
            "bottler_name_addr": "Harborview Imports, Speyside, Scotland",
            "country_of_origin": "Product of Scotland",
            "government_warning": DEFAULT_WARNING,
        },
        "expected": {
            "brand_name": "Harborview Reserve",
            "class_type": "Blended Scotch Whisky",
            "alcohol_content": "43%",
            "net_contents": "750 mL",
            "bottler_name_addr": "Harborview Imports, Speyside, Scotland",
            "country_of_origin": "Product of Scotland",
            "government_warning": DEFAULT_WARNING,
        },
        "context": {"is_import": True, "abv_required": True},
        "warning_style": "compliant",
    },
    {
        "name": "crestline_vodka_bad_warning",
        "description": "Warning rendered in title case, not all-caps/bold -- must FAIL.",
        "actual": {
            "brand_name": "CRESTLINE",
            "class_type": "Vodka",
            "alcohol_content": "40% Alc./Vol.",
            "net_contents": "1 L",
            "bottler_name_addr": "Crestline Spirits, Austin, TX",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING.title(),
        },
        "expected": {
            "brand_name": "Crestline",
            "class_type": "Vodka",
            "alcohol_content": "40%",
            "net_contents": "1 L",
            "bottler_name_addr": "Crestline Spirits, Austin, TX",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING,
        },
        "context": {"is_import": False, "abv_required": True},
        "warning_style": "title_case",
    },
    {
        "name": "redbridge_rum_abv_mismatch",
        "description": "Stated ABV is outside the 0.3-point tolerance -- must FAIL.",
        "actual": {
            "brand_name": "RED BRIDGE",
            "class_type": "Spiced Rum",
            "alcohol_content": "40% Alc./Vol.",
            "net_contents": "750 mL",
            "bottler_name_addr": "Red Bridge Distillers, New Orleans, LA",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING,
        },
        "expected": {
            "brand_name": "Red Bridge",
            "class_type": "Spiced Rum",
            "alcohol_content": "45%",
            "net_contents": "750 mL",
            "bottler_name_addr": "Red Bridge Distillers, New Orleans, LA",
            "country_of_origin": None,
            "government_warning": DEFAULT_WARNING,
        },
        "context": {"is_import": False, "abv_required": True},
        "warning_style": "compliant",
    },
]


def render_label(profile: dict) -> Image.Image:
    """Render one label as a clean PIL image."""
    img = Image.new("RGB", IMAGE_SIZE, "white")
    draw = ImageDraw.Draw(img)
    actual = profile["actual"]
    y = 60

    def draw_line(text, font, gap=20):
        nonlocal y
        draw.text((60, y), text, fill="black", font=font)
        y += font.size + gap

    draw_line(actual["brand_name"], _load_font(44, bold=True), gap=30)
    draw_line(actual["class_type"], _load_font(28))
    draw_line(actual["alcohol_content"], _load_font(28))
    draw_line(actual["net_contents"], _load_font(28))
    draw_line(actual["bottler_name_addr"], _load_font(22))
    if actual.get("country_of_origin"):
        draw_line(actual["country_of_origin"], _load_font(22))

    y += 30
    _draw_warning(draw, actual["government_warning"], profile["warning_style"], y)

    return img


def _draw_wrapped(draw, text, font, x, y, max_width, line_height=30):
    words, line = text.split(), ""
    for word in words:
        test_line = f"{line} {word}".strip()
        if draw.textlength(test_line, font=font) > max_width:
            draw.text((x, y), line, fill="black", font=font)
            y += line_height
            line = word
        else:
            line = test_line
    if line:
        draw.text((x, y), line, fill="black", font=font)


def _draw_warning(draw, text, style, y):
    """
    Draws the 'GOVERNMENT WARNING:' prefix on its own bold line, then the
    body wrapped in regular text below -- matching how real labels only
    bold the lead-in, not the whole statement (per Jenny's description).
    """
    if style == "title_case":
        prefix, prefix_font = "Government Warning:", _load_font(20)
    else:
        prefix, prefix_font = "GOVERNMENT WARNING:", _load_font(20, bold=True)

    draw.text((60, y), prefix, fill="black", font=prefix_font)
    y += prefix_font.size + 10

    body = text.split(":", 1)[1].strip() if ":" in text else text
    _draw_wrapped(draw, body, _load_font(20), x=60, y=y, max_width=780)


def degrade(img: Image.Image) -> Image.Image:
    """
    Simulate a messy real-world photo: skew, blur, dimmer/flatter contrast,
    and a glare patch -- the failure modes Jenny described directly
    (angled shots, bad lighting, glare).
    """
    img = img.rotate(random.uniform(-6, 6), expand=True, fillcolor="white")
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.0)))
    img = ImageEnhance.Brightness(img).enhance(0.8)
    img = ImageEnhance.Contrast(img).enhance(0.75)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    w, h = img.size
    gx, gy = random.randint(0, w // 2), random.randint(0, h // 2)
    odraw.ellipse([gx, gy, gx + w // 3, gy + h // 4], fill=(255, 255, 255, 120))
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=30))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def generate_all():
    # Fixed seed: degrade() uses random rotation/blur/glare with no seed
    # anywhere, so every run previously produced DIFFERENT images under
    # the SAME filenames -- silently invalidating any tuning, debugging,
    # or test expectations built against a prior run. This is almost
    # certainly what made today's results diverge sharply from last
    # session's despite no changes to the detection logic itself.
    random.seed(42)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    applications = {}

    for profile in PROFILES:
        clean_img = render_label(profile)
        for messy, img in (("clean", clean_img), ("messy", degrade(clean_img.copy()))):
            filename = f"{profile['name']}_{messy}.jpg"
            path = os.path.join(OUTPUT_DIR, filename)
            img.convert("RGB").save(path, quality=85)
            expected = {**profile["expected"], "government_warning_header": "GOVERNMENT WARNING:"}
            applications[filename] = {
                "expected": expected,
                "context": profile["context"],
            }
            print(f"Wrote {path}  ({profile['description']})")

    # Deliberately corrupt file -- exercises the "could not load image" path.
    corrupt_path = os.path.join(OUTPUT_DIR, "corrupted_upload.jpg")
    with open(corrupt_path, "wb") as f:
        f.write(b"not a real jpeg file")
    applications["corrupted_upload.jpg"] = {"expected": {}, "context": {}}
    print(f"Wrote {corrupt_path}  (deliberately corrupt -- tests error handling)")

    apps_path = os.path.join(OUTPUT_DIR, "applications.json")
    with open(apps_path, "w") as f:
        json.dump(applications, f, indent=2)
    print(f"\nWrote {apps_path}")
    print(f"\nRun with:\n  python main.py --labels {OUTPUT_DIR}/ --applications {apps_path}")


if __name__ == "__main__":
    generate_all()