"""
Field validation engine.

Validation rules live as data in a small SQLite table (rules/label_rules.db),
not hardcoded in Python -- this keeps the ruleset auditable and editable
without touching code.
"""

import re
import sqlite3
from dataclasses import dataclass

from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

DEFAULT_DB_PATH = "rules/label_rules.db"

_WARNING_HEADER_RE = re.compile(r"^government\s+warning:?\s*", re.IGNORECASE)
_ML_PER_UNIT = {
    "ml": 1.0, "l": 1000.0, "cl": 10.0,
    "floz": 29.5735, "gal": 3785.41,
}
_VOLUME_RE = re.compile(
    r"(\d+\.?\d*)\s*(mL|ml|L|l|cL|cl|fl\.?\s?oz\.?|gal(?:lons?)?)", re.IGNORECASE
)


def _parse_volume_ml(text: str) -> float | None:
    """Parse a volume string like '750 mL' or '25.4 fl oz' into milliliters."""
    match = _VOLUME_RE.search(text)
    if not match:
        return None
    unit = re.sub(r"[.\s]", "", match.group(2)).lower()
    factor = _ML_PER_UNIT.get(unit)
    return float(match.group(1)) * factor if factor else None


def _strip_warning_header(text: str) -> str:
    """Normalize away a leading 'GOVERNMENT WARNING:' so wording comparison
    works whether or not either side happens to include the header."""
    return _WARNING_HEADER_RE.sub("", text.strip())


@dataclass
class FieldResult:
    field: str
    status: str            # "pass", "fail", "not_applicable"
    extracted: str | bool | None
    expected: str | None
    reason: str


def build_rules_db(db_path: str = DEFAULT_DB_PATH) -> None:
    """One-time setup: define the TTB field validation ruleset as data."""
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS field_rules (
            field_name TEXT PRIMARY KEY,
            match_type TEXT NOT NULL,       -- 'fuzzy', 'exact', 'numeric_tolerance', 'boolean_required'
            threshold REAL,                 -- fuzzy score cutoff, or numeric tolerance
            required INTEGER NOT NULL,      -- 1 = always required, 0 = conditional
            condition_field TEXT,           -- e.g. 'is_import', for conditional fields
            fuzzy_method TEXT                -- 'token_sort' (default) or 'token_set'
        )
    """)
    # This function defines the CURRENT ruleset exactly -- it's not an
    # incremental migration. Clear existing rows first: INSERT OR REPLACE
    # below only touches rows present in `rules`, so without this, a
    # renamed or removed field's old row lingers forever and silently
    # resurfaces in validate_record() results (which is exactly what
    # happened with government_warning_formatting_ok after it was split
    # into government_warning_header / government_warning_header_bold).
    conn.execute("DELETE FROM field_rules")
    rules = [
        ("brand_name",                       "fuzzy",                90.0, 1, None,           "token_sort"),
        ("class_type",                       "fuzzy",                85.0, 1, None,           "token_sort"),
        ("alcohol_content",                  "numeric_tolerance",     0.3, 0, "abv_required", None),
        ("net_contents",                     "volume_tolerance",      5.0, 1, None,           None),
        ("bottler_name_addr",                "fuzzy",                80.0, 1, None,           "token_set"),
        ("country_of_origin",                "fuzzy",                85.0, 0, "is_import",    "token_set"),
        ("government_warning",               "exact",                None, 1, None,           None),
        ("government_warning_header",        "exact_case_sensitive", None, 1, None,           None),
        ("government_warning_header_bold",   "boolean_required",     None, 1, None,           None),
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO field_rules VALUES (?, ?, ?, ?, ?, ?)", rules
    )
    conn.commit()
    conn.close()


def load_rules(db_path: str = DEFAULT_DB_PATH) -> dict:
    """Load the field_rules table into a dict keyed by field_name."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM field_rules").fetchall()
    conn.close()
    return {row["field_name"]: dict(row) for row in rows}


def validate_field(field_name: str, extracted: str | bool | None, expected: str | None,
                    rule: dict, context: dict) -> FieldResult:
    """Apply one field's rule and return a human-readable result."""
    if rule["required"] == 0 and rule["condition_field"] and not context.get(rule["condition_field"], False):
        return FieldResult(
            field_name,
            "not_applicable",
            extracted,
            expected,
            f"{field_name} not required for this submission."
        )

    if rule["match_type"] == "boolean_required":
        if extracted is True:
            passed, reason = True, "Bold formatting requirement met."
        elif extracted is False:
            passed, reason = False, "Header text is not rendered in bold."
        else:
            passed, reason = False, "Bold formatting could not be verified from the image."
        return FieldResult(field_name, "pass" if passed else "fail", extracted, expected, reason)

    if extracted is None:
        return FieldResult(field_name, "fail", extracted, expected,
                            f"{field_name} could not be extracted from the label.")

    # Defensive: the model's JSON is only guaranteed to parse, not to be
    # correctly typed per field -- a bare number or list must not crash
    # validation. Coerce rather than fail hard.
    if not isinstance(extracted, str):
        extracted = str(extracted)
    if expected is not None and not isinstance(expected, str):
        expected = str(expected)

    if rule["match_type"] == "exact":
        norm_extracted = re.sub(r"\s+", " ", _strip_warning_header(extracted)).lower()
        norm_expected = re.sub(r"\s+", " ", _strip_warning_header(expected or "")).lower()
        passed = norm_extracted == norm_expected
        reason = ("Body wording matches (header/case/whitespace normalized)." if passed
                  else "Extracted wording does not match the required statement.")

    elif rule["match_type"] == "fuzzy":
        scorer = (fuzz.token_set_ratio if rule.get("fuzzy_method") == "token_set"
                  else fuzz.token_sort_ratio)
        score = scorer(extracted, expected or "", processor=default_process)
        passed = score >= rule["threshold"]
        reason = f"Fuzzy match score {score:.0f} (threshold {rule['threshold']:.0f})."

    elif rule["match_type"] == "exact_case_sensitive":
        norm_extracted = re.sub(r"\s+", " ", extracted.strip())
        norm_expected = re.sub(r"\s+", " ", (expected or "").strip())
        passed = norm_extracted == norm_expected
        reason = ("Exact match, including case." if passed
                  else f"Does not exactly match required text/casing "
                       f"(expected {norm_expected!r}, got {norm_extracted!r}).")

    elif rule["match_type"] == "numeric_tolerance":
        e_match = re.search(r"\d+\.?\d*", extracted)
        x_match = re.search(r"\d+\.?\d*", expected or "")
        if e_match and x_match:
            e_val, x_val = float(e_match.group()), float(x_match.group())
            diff = abs(e_val - x_val)
            passed = diff <= (rule["threshold"] or 0)
            reason = (f"Extracted value {e_val} vs expected {x_val} "
                    f"(difference {diff:.2f}, tolerance {rule['threshold']}).")
        else:
            passed = False
            reason = "Could not find a numeric value in the extracted or expected text."

    elif rule["match_type"] == "volume_tolerance":
        e_val = _parse_volume_ml(extracted)
        x_val = _parse_volume_ml(expected or "")
        if e_val is not None and x_val is not None:
            diff = abs(e_val - x_val)
            passed = diff <= (rule["threshold"] or 0)
            reason = (f"Extracted {e_val:.1f} mL vs expected {x_val:.1f} mL "
                      f"(difference {diff:.1f} mL, tolerance {rule['threshold']} mL).")
        else:
            passed = False
            reason = "Could not parse a recognized volume unit from the extracted or expected text."

    else:
        passed = False
        reason = f"Unknown match type '{rule['match_type']}'."

    return FieldResult(field_name, "pass" if passed else "fail", extracted, expected, reason)


def validate_record(extracted_fields: dict, expected_fields: dict, context: dict,
                     db_path: str = DEFAULT_DB_PATH) -> list:
    """Validate every rule in the ruleset against one application's data."""
    rules = load_rules(db_path)
    results = []
    for field_name, rule in rules.items():
        results.append(validate_field(
            field_name,
            extracted_fields.get(field_name),
            expected_fields.get(field_name),
            rule,
            context,
        ))
    return results