"""Tests for Issue and ChangeLogEntry data models."""

import pytest
from datafix.models import Issue, ChangeLogEntry


def test_issue_creation_and_dict():
    issue = Issue(
        issue_id="ISSUE_001",
        issue_type="missing_value",
        columns=["email"],
        row_ids=[1, 2, 5],
        sample_values=[None, "N/A", ""],
        severity="high",
        description="Found 3 missing email values",
        detector="missing_detector",
        evidence={"disguised_null_count": 2, "null_ratio": 0.05}
    )
    assert issue.issue_id == "ISSUE_001"
    assert issue.severity == "high"
    assert len(issue.row_ids) == 3
    
    d = issue.to_dict()
    assert d["issue_type"] == "missing_value"
    
    restored = Issue.from_dict(d)
    assert restored.issue_id == issue.issue_id
    assert restored.evidence["null_ratio"] == 0.05


def test_issue_invalid_severity():
    with pytest.raises(ValueError, match="Invalid severity"):
        Issue(
            issue_id="ISSUE_002",
            issue_type="invalid_format",
            columns=["phone"],
            row_ids=[0],
            sample_values=["123"],
            severity="critical",  # invalid
            description="desc",
            detector="format_detector"
        )


def test_changelog_creation_and_validation():
    entry = ChangeLogEntry(
        change_id="CHG_001",
        timestamp="2026-11-19T12:00:00Z",
        row_id=10,
        column="country",
        old_value="USA",
        new_value="United States",
        issue_id="ISSUE_001",
        strategy="canonical_mapping",
        reason="Standardized country abbreviation to canonical form",
        confidence=0.95,
        status="applied"
    )
    assert entry.status == "applied"
    assert entry.confidence == 0.95
    
    d = entry.to_dict()
    restored = ChangeLogEntry.from_dict(d)
    assert restored.change_id == entry.change_id
    assert restored.new_value == "United States"


def test_changelog_invalid_status_and_confidence():
    with pytest.raises(ValueError, match="Invalid status"):
        ChangeLogEntry(
            change_id="CHG_002",
            timestamp="2026-11-19T12:00:00Z",
            row_id=1,
            column="col",
            old_value="a",
            new_value="b",
            issue_id="ISSUE_001",
            strategy="strat",
            reason="reason",
            confidence=0.8,
            status="unknown_status"
        )
    
    with pytest.raises(ValueError, match="Confidence score"):
        ChangeLogEntry(
            change_id="CHG_003",
            timestamp="2026-11-19T12:00:00Z",
            row_id=1,
            column="col",
            old_value="a",
            new_value="b",
            issue_id="ISSUE_001",
            strategy="strat",
            reason="reason",
            confidence=1.5,
            status="applied"
        )
