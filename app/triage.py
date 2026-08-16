"""
Triage / routing logic.

Assesses image quality using fast, deterministic signals (no OCR, no
network calls) and decides whether a submission goes to the fast path
or the careful path. Target: well under 100ms per image.
"""

import cv2


def score_image_quality(image_path: str) -> dict:
    """
    Compute quality signals for a label image and decide routing.

    Returns a dict with:
      - readable: bool, whether the image could be loaded at all
      - route: "fast" or "careful" (only present if readable)
      - signals: raw quality metrics, useful for tuning thresholds
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"readable": False, "reason": "could_not_load_image"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur: variance of the Laplacian. Higher = sharper.
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

    # Brightness / contrast
    mean_brightness = gray.mean()
    contrast = gray.std()

    # Resolution check (label text needs enough pixels to OCR reliably)
    height, width = gray.shape
    min_dimension = min(height, width)

    signals = {
        "blur_score": round(float(blur_score), 1),
        "mean_brightness": round(float(mean_brightness), 1),
        "contrast": round(float(contrast), 1),
        "min_dimension": int(min_dimension),
    }

    # Thresholds tuned against sample_labels/ (synthetic renders degraded by
    # generate_test_labels.py's own filters -- revisit against real
    # photographs once they exist; these numbers may not generalize).
    #
    # Brightness upper bound removed: label art sits on a near-white
    # background, so "clean" reads consistently bright (~250) rather than
    # the moderate range a natural photo would suggest. A high ceiling here
    # was actually rejecting every clean label. The lower bound still
    # catches genuine underexposure.
    is_clean = (
        blur_score > 100
        and mean_brightness > 40
        and contrast > 25
        and min_dimension >= 600
    )

    return {
        "readable": True,
        "route": "fast" if is_clean else "careful",
        "signals": signals,
    }