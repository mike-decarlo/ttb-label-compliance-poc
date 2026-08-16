"""
Formats validation results into plain-language output -- built for the
audience described in stakeholder interviews (a wide range of tech
comfort levels), not for developers.
"""


def summarize(results: list) -> dict:
    """Roll up field-level results into an overall decision + explanation."""
    failed = [r for r in results if r.status == "fail"]
    overall = "flag_for_review" if failed else "approve"

    return {
        "overall": overall,
        "fields": [
            {
                "field": r.field,
                "status": r.status,
                "reason": r.reason,
                "extracted": r.extracted,
                "expected": r.expected,
            }
            for r in results
        ],
    }


def to_plain_text(summary: dict) -> str:
    """Human-readable version for a simple UI or CLI output."""
    lines = [
        f"Overall: {summary['overall'].replace('_', ' ').title()}",
        f"Route: {summary.get('route', 'n/a')}",
        "",
    ]
    if "quality_signals" in summary:
        lines.append(f"Quality signals: {summary['quality_signals']}")
    marker_map = {"pass": "OK", "fail": "FLAG", "not_applicable": "N/A"}
    for f in summary["fields"]:
        line = f"[{marker_map[f['status']]}] {f['field']}: {f['reason']}"
        if f["status"] == "fail":
            line += f"\n       extracted={f['extracted']!r}  expected={f['expected']!r}"
        lines.append(line)
    return "\n".join(lines)