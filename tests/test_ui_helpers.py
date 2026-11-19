"""Unit and smoke tests for datafix.ui_helpers."""

import io
import pytest
import pandas as pd
from datafix.config import Config
from datafix.models import Issue, ChangeLogEntry
from datafix.audit import AuditLog
from datafix.pipeline import DataFixPipeline
from datafix.ui_helpers import (
    validate_and_load_file,
    filter_issues,
    apply_change_decisions,
    apply_bulk_threshold,
    generate_export_bundle,
)


def test_validate_and_load_file_empty():
    """Test that empty inputs are rejected with clear messages."""
    # None
    df, err = validate_and_load_file(None)
    assert df is None
    assert "No file provided" in err

    # Empty StringIO
    empty_buf = io.StringIO("")
    empty_buf.name = "empty.csv"
    df, err = validate_and_load_file(empty_buf)
    assert df is None
    assert "empty" in err.lower() or "no columns" in err.lower()


def test_validate_and_load_file_single_column():
    """Test that single-column files are rejected."""
    buf = io.StringIO("col_a\n1\n2\n3\n")
    buf.name = "single.csv"
    df, err = validate_and_load_file(buf)
    assert df is None
    assert "single column" in err.lower()


def test_validate_and_load_file_valid_csv():
    """Test loading a valid multi-column CSV."""
    buf = io.StringIO("name,age,city\nAlice,30,New York\nBob,25,Chicago\n")
    buf.name = "valid.csv"
    df, err = validate_and_load_file(buf)
    assert err is None
    assert df is not None
    assert "_row_id" in df.columns
    assert len(df) == 2


def test_validate_and_load_file_encoding_fallback():
    """Test CSV with non-UTF8 encoding (latin-1)."""
    # Create bytes with latin1 character 'é' = 0xe9
    latin1_bytes = "name,location\nRenée,Montréal\n".encode("latin1")
    buf = io.BytesIO(latin1_bytes)
    buf.name = "latin1.csv"
    df, err = validate_and_load_file(buf)
    assert err is None
    assert df is not None
    assert len(df) == 1
    assert "_row_id" in df.columns


def test_filter_issues():
    """Test filtering issues by type and severity."""
    i1 = Issue(
        issue_id="ISS_1",
        issue_type="missing_value",
        columns=["age"],
        row_ids=[1],
        sample_values=[None],
        severity="high",
        description="Missing age",
        detector="missing"
    )
    i2 = Issue(
        issue_id="ISS_2",
        issue_type="invalid_format",
        columns=["date"],
        row_ids=[2],
        sample_values=["bad"],
        severity="low",
        description="Bad date",
        detector="format"
    )
    issues = [i1, i2]

    # Filter by type
    assert len(filter_issues(issues, issue_type="missing_value")) == 1
    assert filter_issues(issues, issue_type="missing_value")[0].issue_id == "ISS_1"

    # Filter by severity
    assert len(filter_issues(issues, severity="low")) == 1
    assert filter_issues(issues, severity="low")[0].issue_id == "ISS_2"

    # Filter with All
    assert len(filter_issues(issues, issue_type="All", severity="All")) == 2


def test_apply_change_decisions():
    """Test updating decisions and reconstructing the DataFrame."""
    df_orig = pd.DataFrame({
        "_row_id": [1, 2],
        "name": ["Alice", "Bob"],
        "email": ["alice@example.com", "bad-email"]
    })
    entry = ChangeLogEntry(
        change_id="CHG_001",
        timestamp="2026-01-01T00:00:00Z",
        row_id=2,
        column="email",
        old_value="bad-email",
        new_value="bob@example.com",
        issue_id="ISS_1",
        strategy="format_email",
        reason="Formatted email",
        confidence=0.9,
        status="applied"
    )
    audit = AuditLog([entry])

    # Reject the change
    repaired, updated_audit = apply_change_decisions(df_orig, audit, {"CHG_001": "rejected"})
    assert updated_audit.get_by_id("CHG_001").status == "rejected"
    assert repaired.loc[repaired["_row_id"] == 2, "email"].values[0] == "bad-email"

    # Approve the change
    repaired2, updated_audit2 = apply_change_decisions(df_orig, audit, {"CHG_001": "approved"})
    assert updated_audit2.get_by_id("CHG_001").status == "approved"
    assert repaired2.loc[repaired2["_row_id"] == 2, "email"].values[0] == "bob@example.com"


def test_apply_bulk_threshold():
    """Test bulk approval of suggestions above threshold."""
    e1 = ChangeLogEntry(
        change_id="C1",
        timestamp="2026-01-01T00:00:00Z",
        row_id=1,
        column="col",
        old_value="a",
        new_value="b",
        issue_id="I1",
        strategy="strat",
        reason="reason",
        confidence=0.88,
        status="suggested"
    )
    e2 = ChangeLogEntry(
        change_id="C2",
        timestamp="2026-01-01T00:00:00Z",
        row_id=2,
        column="col",
        old_value="c",
        new_value="d",
        issue_id="I2",
        strategy="strat",
        reason="reason",
        confidence=0.72,
        status="suggested"
    )
    audit = AuditLog([e1, e2])

    count = apply_bulk_threshold(audit, threshold=0.85)
    assert count == 1
    assert e1.status == "approved"
    assert e2.status == "suggested"


def test_generate_export_bundle():
    """Test generating full export package in memory."""
    df = pd.DataFrame({
        "customer_id": [1, 2],
        "name": ["Alice", "Bob"],
        "email": ["alice@example.com", "bob@example.com"]
    })
    pipeline = DataFixPipeline()
    result = pipeline.run(df, dataset_name="TestBundle")

    bundle = generate_export_bundle(
        original_df=result.original_df,
        repaired_df=result.repaired_df,
        audit_log=result.audit_log,
        issues_before=result.issues_before,
        diagnoses=result.diagnoses,
        dataset_name="TestBundle"
    )

    assert "cleaned_csv" in bundle
    assert "customer_id,name,email" in bundle["cleaned_csv"]
    assert "_row_id" not in bundle["cleaned_csv"]
    assert "quality_report_html" in bundle
    assert "<!DOCTYPE html>" in bundle["quality_report_html"]
    assert "change_log_csv" in bundle
    assert "change_log_json" in bundle
    assert "validation" in bundle


def test_paginate_entries():
    """Test pagination of audit entries."""
    from datafix.ui_helpers import paginate_entries
    entries = [
        ChangeLogEntry(
            change_id=f"C_{i}",
            timestamp="2026-01-01T00:00:00Z",
            row_id=i,
            column="col",
            old_value="a",
            new_value="b",
            issue_id="I1",
            strategy="strat",
            reason="reason",
            confidence=0.9,
            status="applied"
        )
        for i in range(450)
    ]
    p1, total = paginate_entries(entries, page=1, page_size=200)
    assert len(p1) == 200
    assert total == 3
    assert p1[0].change_id == "C_0"

    p2, _ = paginate_entries(entries, page=2, page_size=200)
    assert len(p2) == 200
    assert p2[0].change_id == "C_200"

    p3, _ = paginate_entries(entries, page=3, page_size=200)
    assert len(p3) == 50
    assert p3[0].change_id == "C_400"


def test_build_diff_dataframe_and_styler():
    """Test creating formatted diff dataframe and confidence styling."""
    from datafix.ui_helpers import build_diff_dataframe, style_confidence_dataframe
    entries = [
        ChangeLogEntry("C1", "2026-01-01", 1, "col", "old1", "new1", "I1", "strat", "reason1", 0.95, "applied"),
        ChangeLogEntry("C2", "2026-01-01", 2, "col", "old2", "new2", "I2", "strat", "reason2", 0.75, "suggested"),
        ChangeLogEntry("C3", "2026-01-01", 3, "col", "old3", "new3", "I3", "strat", "reason3", 0.40, "suggested"),
    ]
    df = build_diff_dataframe(entries)
    assert len(df) == 3
    assert "diff" in df.columns
    assert "old1  ➔  new1" in df["diff"].iloc[0]
    assert df["confidence_tier"].iloc[0] == "High (≥0.85)"
    assert df["confidence_tier"].iloc[1] == "Medium (0.60–0.85)"
    assert df["confidence_tier"].iloc[2] == "Low (<0.60)"

    styler = style_confidence_dataframe(df)
    assert styler is not None


def test_compute_projected_threshold_impact():
    """Test what-if projected impact calculation across thresholds."""
    from datafix.ui_helpers import compute_projected_threshold_impact
    df = pd.DataFrame({"col": [1, 2, 3, 4]})
    issue1 = Issue("I1", "outlier", ["col"], [1], [999], "high", "outlier", "outlier")
    issue2 = Issue("I2", "missing_value", ["col"], [2], [None], "med", "missing", "missing")
    
    e1 = ChangeLogEntry("C1", "2026-01-01", 1, "col", 999, 10, "I1", "strat", "reason", 0.90, "applied")
    e2 = ChangeLogEntry("C2", "2026-01-01", 2, "col", None, 5, "I2", "strat", "reason", 0.70, "suggested")
    audit = AuditLog([e1, e2])

    # Threshold 0.85: only e1 applied
    impact_85 = compute_projected_threshold_impact(df, [issue1, issue2], audit, threshold=0.85)
    assert impact_85["applied_count"] == 1
    assert impact_85["suggested_count"] == 1
    assert impact_85["resolved_issues_count"] == 1

    # Threshold 0.65: both e1 and e2 applied
    impact_65 = compute_projected_threshold_impact(df, [issue1, issue2], audit, threshold=0.65)
    assert impact_65["applied_count"] == 2
    assert impact_65["suggested_count"] == 0
    assert impact_65["resolved_issues_count"] == 2
    assert impact_65["projected_score"] >= impact_85["projected_score"]


