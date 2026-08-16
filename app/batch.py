"""
Concurrent batch orchestration -- so a 200-300 item batch (per the
importer workflow described in interviews) isn't processed serially.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from app.extraction import careful_extract, fast_extract
from app.reporting import summarize
from app.triage import score_image_quality
from app.validation import validate_record


def process_single_application(image_path: str, expected_fields: dict,
                                context: dict) -> dict:
    """Full pipeline for one submission: triage -> extract -> validate -> report."""
    quality = score_image_quality(image_path)
    if not quality.get("readable"):
        return {
            "label": image_path,
            "overall": "flag_for_review",
            "reason": "Image could not be opened -- please re-upload.",
        }

    extracted = (fast_extract(image_path) if quality["route"] == "fast"
                 else careful_extract(image_path))

    results = validate_record(extracted, expected_fields, context)
    summary = summarize(results)
    summary["label"] = image_path
    summary["route"] = quality["route"]
    summary["quality_signals"] = quality.get("signals")
    return summary


def process_batch(label_paths: list, application_data: dict, context_data: dict,
                   max_workers: int = 8) -> list:
    """
    Route and validate a batch concurrently. `application_data` and
    `context_data` are keyed by label_path, matching each image to its
    expected fields and context flags (is_import, abv_required, etc).
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                process_single_application,
                path,
                application_data[path],
                context_data.get(path, {}),
            ): path
            for path in label_paths
        }
        for future in as_completed(futures):
            path = futures[future]
            try:
                results.append(future.result())
            except Exception as e:  # noqa: BLE001
                results.append({"label": path, "overall": "error", "reason": str(e)})
    return results