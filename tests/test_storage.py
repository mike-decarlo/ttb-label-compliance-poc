"""Tests for app/storage.py -- result persistence."""

from app.storage import load_results, save_result


def test_save_and_load_result(tmp_path):
    db_path = str(tmp_path / "test_results.db")
    result = {
        "label": "sample_labels/old_tom_bourbon_clean.jpg",
        "route": "fast",
        "overall": "approve",
        "fields": [{
            "field": "brand_name", "status": "pass",
            "reason": "Fuzzy match score 100 (threshold 90).",
            "extracted": "OLD TOM DISTILLERY", "expected": "OLD TOM DISTILLERY",
        }],
    }
    save_result(result, db_path=db_path)

    loaded = load_results(db_path=db_path)
    assert len(loaded) == 1
    assert loaded[0]["label_path"] == result["label"]
    assert loaded[0]["overall"] == "approve"
    assert loaded[0]["fields"][0]["field"] == "brand_name"


def test_save_result_handles_missing_fields_key(tmp_path):
    """The corrupted-image / batch-error result shapes don't include a
    'fields' key at all -- save_result must not crash on that."""
    db_path = str(tmp_path / "test_results.db")
    result = {"label": "sample_labels/corrupted_upload.jpg", "overall": "flag_for_review",
               "reason": "Image could not be opened -- please re-upload."}
    save_result(result, db_path=db_path)
    loaded = load_results(db_path=db_path)
    assert loaded[0]["fields"] == []


def test_load_results_orders_most_recent_first(tmp_path):
    db_path = str(tmp_path / "test_results.db")
    save_result({"label": "first.jpg", "overall": "approve"}, db_path=db_path)
    save_result({"label": "second.jpg", "overall": "approve"}, db_path=db_path)
    loaded = load_results(db_path=db_path)
    assert loaded[0]["label_path"] == "second.jpg"


def test_load_results_respects_limit(tmp_path):
    db_path = str(tmp_path / "test_results.db")
    for i in range(5):
        save_result({"label": f"label{i}.jpg", "overall": "approve"}, db_path=db_path)
    loaded = load_results(db_path=db_path, limit=2)
    assert len(loaded) == 2