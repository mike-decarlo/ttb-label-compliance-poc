"""
Command-line entry point for the TTB label compliance POC.

Usage:
    python main.py --labels sample_labels/ --applications applications.json
    python main.py --labels sample_labels/old_tom.jpg --applications applications.json
    python main.py --labels sample_labels/ --applications applications.json --output report.json
"""

import argparse
import json
import os

from app.batch import process_batch, process_single_application
from app.reporting import to_plain_text


def load_applications(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Run TTB label compliance checks.")
    parser.add_argument("--labels", required=True,
                         help="Path to a single label image or a directory of images.")
    parser.add_argument("--applications", required=True,
                         help="JSON file mapping filenames to expected fields + context.")
    parser.add_argument("--output", default=None,
                         help="Optional path to write a JSON report instead of printing.")
    parser.add_argument("--max-workers", type=int, default=8,
                         help="Concurrent worker count for batch processing (default: 8).")
    args = parser.parse_args()

    applications = load_applications(args.applications)

    if os.path.isdir(args.labels):
        found = {
            fname: os.path.join(args.labels, fname)
            for fname in applications
            if os.path.exists(os.path.join(args.labels, fname))
        }
        missing = [fname for fname in applications if fname not in found]

        label_paths = list(found.values())
        application_data = {p: applications[os.path.basename(p)]["expected"] for p in label_paths}
        context_data = {p: applications[os.path.basename(p)].get("context", {}) for p in label_paths}
        results = process_batch(label_paths, application_data, context_data,
                                 max_workers=args.max_workers)

        for fname in missing:
            results.append({
                "label": os.path.join(args.labels, fname),
                "overall": "flag_for_review",
                "reason": "Listed in applications.json but the file was not found in --labels.",
            })
    else:
        fname = os.path.basename(args.labels)
        entry = applications[fname]
        try:
            results = [process_single_application(
                args.labels, entry["expected"], entry.get("context", {}))]
        except Exception as e:  # noqa: BLE001
            results = [{"label": args.labels, "overall": "error", "reason": str(e)}]

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {len(results)} result(s) to {args.output}")
    else:
        for r in results:
            print(f"\n=== {r.get('label', '?')} ===")
            print(to_plain_text(r) if "fields" in r else r)


if __name__ == "__main__":
    main()