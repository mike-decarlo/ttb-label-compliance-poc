"""
Pins expected routing for the sample_labels fixtures, so a threshold
regression (like the inverted brightness ceiling) can't ship silently
again. Requires sample_labels/ to exist -- run
`python scripts/generate_test_labels.py` first if these are skipped.
"""

import os

import pytest

from app.triage import score_image_quality

SAMPLE_DIR = "sample_labels"

CLEAN_FILES = [
    "old_tom_bourbon_clean.jpg",
    "stones_throw_gin_clean.jpg",
    "harborview_import_whisky_clean.jpg",
    "crestline_vodka_bad_warning_clean.jpg",
    "redbridge_rum_abv_mismatch_clean.jpg",
]

MESSY_FILES = [
    "old_tom_bourbon_messy.jpg",
    "stones_throw_gin_messy.jpg",
    "harborview_import_whisky_messy.jpg",
    "crestline_vodka_bad_warning_messy.jpg",
    "redbridge_rum_abv_mismatch_messy.jpg",
]


def _path(fname):
    return os.path.join(SAMPLE_DIR, fname)


def _skip_if_missing(fname):
    if not os.path.exists(_path(fname)):
        pytest.skip(f"{fname} not found -- run scripts/generate_test_labels.py first")


@pytest.mark.parametrize("fname", CLEAN_FILES)
def test_clean_labels_route_fast(fname):
    _skip_if_missing(fname)
    result = score_image_quality(_path(fname))
    assert result["readable"] is True
    assert result["route"] == "fast", f"{fname} signals: {result['signals']}"


@pytest.mark.parametrize("fname", MESSY_FILES)
def test_messy_labels_route_careful(fname):
    _skip_if_missing(fname)
    result = score_image_quality(_path(fname))
    assert result["readable"] is True
    assert result["route"] == "careful", f"{fname} signals: {result['signals']}"


def test_corrupt_file_is_unreadable():
    fname = "corrupted_upload.jpg"
    _skip_if_missing(fname)
    result = score_image_quality(_path(fname))
    assert result["readable"] is False