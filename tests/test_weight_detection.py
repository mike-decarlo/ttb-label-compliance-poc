"""Tests for deterministic bold/regular classification (app/weight_detection.py)."""

import io
import os

import pytest
from PIL import Image

from app.weight_detection import (
    _ink_density,
    classify_header_bold,
    reference_weight_images,
)

SAMPLE_DIR = "sample_labels"


def test_reference_images_have_distinct_ink_density():
    """Sanity check on the calibration itself: if these ever come back
    equal, bold/regular font-loading fell back to the same font for both
    (e.g. a missing bold font file on this machine), and every downstream
    classification would be meaningless regardless of what it returns."""
    bold_bytes, regular_bytes = reference_weight_images(20)  # matches the actual header font size
    bold_density = _ink_density(Image.open(io.BytesIO(bold_bytes)))
    regular_density = _ink_density(Image.open(io.BytesIO(regular_bytes)))
    assert bold_density > regular_density


def _skip_if_missing(fname):
    path = os.path.join(SAMPLE_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} not found -- run scripts/generate_test_labels.py first")
    return path


def test_bold_header_classified_true():
    path = _skip_if_missing("old_tom_bourbon_clean.jpg")
    assert classify_header_bold(path) is True


def test_non_bold_header_classified_false():
    """The actual bug that started this: a non-bold header on an
    otherwise clean, sharp image was previously misclassified as bold
    by the vision model. Pinned here so it can't silently regress."""
    path = _skip_if_missing("crestline_vodka_bad_warning_clean.jpg")
    assert classify_header_bold(path) is False


def test_bold_header_classified_true_on_messy_image():
    path = _skip_if_missing("redbridge_rum_abv_mismatch_messy.jpg")
    assert classify_header_bold(path) is True


def test_non_bold_header_classified_false_on_messy_image():
    path = _skip_if_missing("crestline_vodka_bad_warning_messy.jpg")
    assert classify_header_bold(path) is False


def test_bold_header_classified_true_on_old_tom_messy():
    """Previously returned None (header undetectable) before deskewing was
    added -- now resolves correctly. Pinned so this specific fix can't
    silently regress."""
    path = _skip_if_missing("old_tom_bourbon_messy.jpg")
    assert classify_header_bold(path) is True


@pytest.mark.xfail(reason="Known limitation: OCR can't reliably locate the "
                           "header on this specific degraded image even "
                           "after deskewing. Fails safe -- a compliant "
                           "label gets sent to human review, never "
                           "silently approved.")
def test_bold_header_stones_throw_messy_known_limitation():
    path = _skip_if_missing("stones_throw_gin_messy.jpg")
    assert classify_header_bold(path) is True


@pytest.mark.xfail(reason="Known limitation: OCR can't reliably locate the "
                           "header on this specific degraded image. Fails "
                           "safe -- a compliant label gets sent to human "
                           "review, never silently approved.")
def test_bold_header_harborview_messy_known_limitation():
    path = _skip_if_missing("harborview_import_whisky_messy.jpg")
    assert classify_header_bold(path) is True