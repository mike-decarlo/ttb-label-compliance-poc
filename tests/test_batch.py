"""
Integration tests for batch.py -- the full pipeline (triage -> extract ->
validate -> report) exercised against real sample images. Slower than
test_validation.py's pure-logic tests, since each assertion triggers live
model calls and/or image processing.

Requires: sample_labels/ generated (scripts/generate_test_labels.py) and
Ollama running locally with qwen2.5:14b and qwen2.5vl:7b pulled.
"""

import json
import os

import pytest

from app.batch import process_batch, process_single_application

SAMPLE_DIR = "sample_labels"


def _skip_if_missing(fname):
    path = os.path.join(SAMPLE_DIR, fname)
    if not os.path.exists(path):
        pytest.skip(f"{fname} not found -- run scripts/generate_test_labels.py first")
    return path


def _load_expected(fname):
    apps_path = os.path.join(SAMPLE_DIR, "applications.json")
    if not os.path.exists(apps_path):
        pytest.skip("applications.json not found -- run scripts/generate_test_labels.py first")
    with open(apps_path) as f:
        apps = json.load(f)
    return apps[fname]["expected"], apps[fname].get("context", {})


def test_process_single_application_approves_clean_compliant_label():
    fname = "old_tom_bourbon_clean.jpg"
    path = _skip_if_missing(fname)
    expected, context = _load_expected(fname)
    result = process_single_application(path, expected, context)
    assert result["overall"] == "approve", result


def test_process_single_application_flags_deliberate_abv_mismatch():
    fname = "redbridge_rum_abv_mismatch_clean.jpg"
    path = _skip_if_missing(fname)
    expected, context = _load_expected(fname)
    result = process_single_application(path, expected, context)
    assert result["overall"] == "flag_for_review"
    alcohol_result = next(f for f in result["fields"] if f["field"] == "alcohol_content")
    assert alcohol_result["status"] == "fail"


def test_corrupted_image_flags_without_crashing():
    fname = "corrupted_upload.jpg"
    path = _skip_if_missing(fname)
    result = process_single_application(path, {}, {})
    assert result["overall"] == "flag_for_review"
    assert "could not be opened" in result["reason"].lower()


def test_process_batch_returns_one_result_per_input_regardless_of_order():
    """ThreadPoolExecutor + as_completed() means result ORDER isn't
    guaranteed to match input order -- compare by filename, not position,
    the same lesson from the determinism check earlier in this project."""
    filenames = ["old_tom_bourbon_clean.jpg", "corrupted_upload.jpg"]
    paths = [_skip_if_missing(f) for f in filenames]

    application_data, context_data = {}, {}
    for path, fname in zip(paths, filenames):
        expected, context = _load_expected(fname)
        application_data[path] = expected
        context_data[path] = context

    results = process_batch(paths, application_data, context_data, max_workers=2)
    result_labels = {r["label"] for r in results}

    assert len(results) == len(paths)
    assert result_labels == set(paths)