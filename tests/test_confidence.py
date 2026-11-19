"""Tests for datafix/confidence.py."""

from datafix.confidence import calculate_repair_confidence
from datafix.models import Issue


def test_confidence_scoring_bounds():
    dummy_issue = Issue(
        issue_id="ISS_01",
        issue_type="invalid_format",
        columns=["price"],
        row_ids=[1],
        sample_values=["$100"],
        severity="med",
        description="desc",
        detector="format_detector"
    )

    # Standardize disguised null: 0.99
    c1 = calculate_repair_confidence(dummy_issue, "standardize_disguised_null", "N/A", None)
    assert 0.98 <= c1 <= 1.0

    # Parse numeric string: 0.98
    c2 = calculate_repair_confidence(dummy_issue, "parse_numeric_string", "$100", 100.0)
    assert 0.95 <= c2 <= 1.0

    # Fuzzy duplicate deduplication: < 0.85 (must be suggestion only)
    c3 = calculate_repair_confidence(dummy_issue, "deduplicate_fuzzy_row", "Alice", "Alicee")
    assert c3 < 0.85

    # Swap inverted dates: < 0.85 (ambiguous)
    c4 = calculate_repair_confidence(dummy_issue, "swap_inverted_dates", "2023-01-02", "2023-01-01")
    assert c4 < 0.85

    # Orphan foreign key: < 0.85
    c5 = calculate_repair_confidence(dummy_issue, "orphan_foreign_key_flag", 9999, 9999)
    assert c5 < 0.85
