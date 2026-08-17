"""Shared font-loading helper -- used by scripts/generate_test_labels.py
for rendering synthetic labels, and by app/extraction.py for rendering
the bold/regular reference images used in the header check."""

import os

from PIL import ImageFont


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Try common system font paths across platforms; fall back to a
    scalable default so a missing font file degrades gracefully instead
    of silently collapsing to an unreadably tiny bitmap font."""
    candidates = [
        f"/usr/share/fonts/truetype/dejavu/DejaVuSans{'-Bold' if bold else ''}.ttf",
        f"/usr/share/fonts/truetype/liberation/LiberationSans-{'Bold' if bold else 'Regular'}.ttf",
        f"/System/Library/Fonts/Supplemental/Arial{' Bold' if bold else ''}.ttf",  # macOS
        "C:\\Windows\\Fonts\\arialbd.ttf" if bold else "C:\\Windows\\Fonts\\arial.ttf",  # Windows
    ]
    for path in candidates:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)  # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()