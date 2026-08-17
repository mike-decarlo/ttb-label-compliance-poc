"""View recently persisted results without re-running the pipeline.

Usage: python scripts/view_results.py [--limit N]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.storage import load_results


def main():
    parser = argparse.ArgumentParser(description="View persisted label review results.")
    parser.add_argument("--limit", type=int, default=20,
                         help="Number of most recent results to show (default: 20).")
    args = parser.parse_args()

    results = load_results(limit=args.limit)
    if not results:
        print("No persisted results found. Run main.py first.")
        return

    for r in results:
        print(f"\n=== {r['label_path']}  ({r['processed_at']}) ===")
        print(f"Overall: {r['overall'].replace('_', ' ').title()}  Route: {r.get('route', 'n/a')}")
        for f in r["fields"]:
            marker = {"pass": "OK", "fail": "FLAG", "not_applicable": "N/A"}.get(f.get("status"), "?")
            print(f"  [{marker}] {f.get('field')}: {f.get('reason')}")


if __name__ == "__main__":
    main()