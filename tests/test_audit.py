"""Tests for datafix/audit.py."""

import pandas as pd
from datafix.audit import AuditLog
from datafix.models import ChangeLogEntry


def test_audit_log_workflow():
    audit = AuditLog()
    entry1 = ChangeLogEntry(
        change_id="CHG_01",
        timestamp="2026-11-19T12:00:00Z",
        row_id=1,
        column="country",
        old_value="USA",
        new_value="United States",
        issue_id="ISS_01",
        strategy="canonical_synonym_mapping",
        reason="reason",
        confidence=0.96,
        status="applied"
    )
    entry2 = ChangeLogEntry(
        change_id="CHG_02",
        timestamp="2026-11-19T12:00:00Z",
        row_id=2,
        column="name",
        old_value="Bob",
        new_value="Bob Smith",
        issue_id="ISS_02",
        strategy="deduplicate_fuzzy_row",
        reason="reason",
        confidence=0.70,
        status="suggested"
    )
    audit.add_entry(entry1)
    audit.add_entry(entry2)

    assert len(audit) == 2
    summary = audit.summary()
    assert summary["applied"] == 1
    assert summary["suggested"] == 1

    # Approve suggestion
    assert audit.approve("CHG_02") is True
    assert audit.get_by_id("CHG_02").status == "approved"

    # Reject
    assert audit.reject("CHG_02") is True
    assert audit.get_by_id("CHG_02").status == "rejected"


def test_audit_revert():
    df = pd.DataFrame({"_row_id": [0, 1], "country": ["Canada", "United States"]})
    audit = AuditLog()
    entry = ChangeLogEntry(
        change_id="CHG_REV",
        timestamp="2026-11-19T12:00:00Z",
        row_id=1,
        column="country",
        old_value="USA",
        new_value="United States",
        issue_id="ISS_01",
        strategy="canonical_synonym_mapping",
        reason="reason",
        confidence=0.96,
        status="applied"
    )
    audit.add_entry(entry)

    # Revert change
    df_reverted = audit.revert("CHG_REV", df)
    assert df.loc[df["_row_id"] == 1, "country"].values[0] == "United States"  # original unchanged
    assert df_reverted.loc[df_reverted["_row_id"] == 1, "country"].values[0] == "USA"  # reverted
    assert entry.status == "reverted"
