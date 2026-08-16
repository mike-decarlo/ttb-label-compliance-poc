"""Basic tests for the validation engine. Run with: pytest"""

import pytest

from app.validation import build_rules_db, load_rules, validate_field


@pytest.fixture
def rules(tmp_path):
    db_path = str(tmp_path / "test_rules.db")
    build_rules_db(db_path)
    return load_rules(db_path)


def test_exact_match_passes_on_identical_text(rules):
    rule = rules["government_warning"]
    result = validate_field(
        "government_warning", "GOVERNMENT WARNING: text here",
        "GOVERNMENT WARNING: text here", rule, context={},
    )
    assert result.status == "pass"


def test_exact_match_tolerates_case_difference(rules):
    """
    Wording match is case/whitespace-insensitive by design -- the
    all-caps/bold *formatting* requirement is checked separately by
    government_warning_formatting_ok, not by this field's wording match.
    """
    rule = rules["government_warning"]
    result = validate_field(
        "government_warning", "Government Warning: text here",
        "GOVERNMENT WARNING: text here", rule, context={},
    )
    assert result.status == "pass"


def test_fuzzy_match_tolerates_case_and_punctuation(rules):
    rule = rules["brand_name"]
    result = validate_field("brand_name", "STONE'S THROW", "Stone's Throw", rule, context={})
    assert result.status == "pass"


def test_numeric_tolerance_within_bounds(rules):
    rule = rules["alcohol_content"]
    result = validate_field(
        "alcohol_content", "44.8% Alc./Vol.", "45% Alc./Vol.", rule,
        context={"abv_required": True},
    )
    assert result.status == "pass"


def test_numeric_tolerance_exceeded(rules):
    rule = rules["alcohol_content"]
    result = validate_field(
        "alcohol_content", "42% Alc./Vol.", "45% Alc./Vol.", rule,
        context={"abv_required": True},
    )
    assert result.status == "fail"


def test_conditional_field_skipped_when_not_applicable(rules):
    rule = rules["country_of_origin"]
    result = validate_field("country_of_origin", None, None, rule, context={"is_import": False})
    assert result.status == "not_applicable"


def test_token_set_fuzzy_tolerates_partial_match(rules):
    """country_of_origin uses token_set, not token_sort -- 'Scotland'
    should match 'Product of Scotland' even though one is a substring
    of the other's wording, not a reordering of it."""
    rule = rules["country_of_origin"]
    result = validate_field("country_of_origin", "Scotland", "Product of Scotland",
                             rule, context={"is_import": True})
    assert result.status == "pass"


def test_type_coercion_prevents_crash_on_non_string_extracted(rules):
    """Model output is only guaranteed to be valid JSON, not correctly
    typed -- a bare number instead of a string must not crash validation."""
    rule = rules["alcohol_content"]
    result = validate_field("alcohol_content", 45.0, "45% Alc./Vol.", rule,
                             context={"abv_required": True})
    assert result.status == "pass"


def test_government_warning_matches_without_header_prefix(rules):
    """extraction.py now asks for the warning body only, without the
    header -- validation must still match it against an expected value
    that includes the header."""
    rule = rules["government_warning"]
    result = validate_field(
        "government_warning",
        "(1) Example warning body text.",
        "GOVERNMENT WARNING: (1) Example warning body text.",
        rule, context={},
    )
    assert result.status == "pass"


def test_volume_tolerance_converts_across_units(rules):
    """750 mL and 0.75 L are the same volume in different units --
    must pass, not fail on differing raw digits."""
    rule = rules["net_contents"]
    result = validate_field("net_contents", "750 mL", "0.75 L", rule, context={})
    assert result.status == "pass"


def test_volume_tolerance_catches_real_unit_mismatch(rules):
    """750 mL and 750 fl oz are NOT the same volume -- must fail, not
    false-pass on matching raw digits with different units."""
    rule = rules["net_contents"]
    result = validate_field("net_contents", "750 mL", "750 fl oz", rule, context={})
    assert result.status == "fail"


def test_volume_tolerance_fails_closed_on_unrecognized_unit(rules):
    rule = rules["net_contents"]
    result = validate_field("net_contents", "750 bottles", "750 mL", rule, context={})
    assert result.status == "fail"


def test_header_exact_match_case_sensitive_passes_on_correct_caps(rules):
    rule = rules["government_warning_header"]
    result = validate_field(
        "government_warning_header", "GOVERNMENT WARNING:", "GOVERNMENT WARNING:",
        rule, context={},
    )
    assert result.status == "pass"


def test_header_exact_match_case_sensitive_fails_on_title_case(rules):
    """The actual TTB requirement Jenny described -- title case is a real
    violation, caught here by comparing literal extracted characters, not
    by trusting a model's true/false self-report."""
    rule = rules["government_warning_header"]
    result = validate_field(
        "government_warning_header", "Government Warning:", "GOVERNMENT WARNING:",
        rule, context={},
    )
    assert result.status == "fail"


def test_header_bold_check_fails_independently_of_wording(rules):
    """Bold-ness and wording/casing are now two separate checks -- a
    failure here should be attributable specifically to bold, not
    conflated with a wording problem."""
    rule = rules["government_warning_header_bold"]
    result = validate_field(
        "government_warning_header_bold", False, None, rule, context={},
    )
    assert result.status == "fail"
    assert "bold" in result.reason.lower()


def test_header_bold_check_fails_when_unknown(rules):
    rule = rules["government_warning_header_bold"]
    result = validate_field(
        "government_warning_header_bold", None, None, rule, context={},
    )
    assert result.status == "fail"