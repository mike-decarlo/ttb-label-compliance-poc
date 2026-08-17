"""
Deskewing: estimates and corrects small rotation angles in a label image
before OCR, via a projection-profile method rather than character-level
detection.

Why this is expected to be more blur-tolerant than OCR itself: OCR needs
to resolve individual letterforms, which heavy blur destroys. This
method only needs row-level ink density patterns -- a whole text line
reads as "darker than the gaps above and below it" -- which is coarser
structure that survives blur much better than fine character detail.
Not guaranteed to succeed on every image, but a fundamentally different
signal than what's already been tried and failed.
"""

import cv2
import numpy as np
from PIL import Image


def estimate_skew_angle(image: Image.Image, angle_range: float = 15.0,
                         step: float = 0.5) -> float:
    """
    Finds the rotation angle that best aligns text rows: tests a range
    of candidate angles and picks the one that makes the horizontal ink
    density profile most "peaky" (tightly banded rows = well-aligned
    text; smeared rows = still skewed).
    """
    gray = np.array(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-angle_range, angle_range + step, step):
        rotated = _rotate(binary, angle)
        row_density = rotated.sum(axis=1)
        score = row_density.var()
        if score > best_score:
            best_score = score
            best_angle = float(angle)
    return best_angle


def deskew(image: Image.Image, angle_range: float = 15.0, step: float = 0.5) -> Image.Image:
    """Estimates the skew angle and returns a corrected copy of the image."""
    angle = estimate_skew_angle(image, angle_range, step)
    gray = np.array(image.convert("L"))
    corrected = _rotate(gray, angle, border_value=255, interpolation=cv2.INTER_LINEAR)
    return Image.fromarray(corrected).convert("RGB")


def _rotate(arr: np.ndarray, angle: float, border_value: int = 0,
            interpolation: int = cv2.INTER_NEAREST) -> np.ndarray:
    h, w = arr.shape
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(arr, matrix, (w, h), flags=interpolation, borderValue=border_value)