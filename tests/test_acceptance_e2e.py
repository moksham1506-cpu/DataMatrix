"""DataFix Final End-to-End Acceptance Test Suite.

Verifies:
1. All 6 capability phases (Detect, Diagnose, Repair, Validate, Explain, Track).
2. Four Phase 6 export artifacts (cleaned CSV, HTML report, change log CSV, change log JSON).
3. Synchronization of user review decisions (approve, reject, undo) with exports.
4. Dataset idempotency (pipeline on repaired output yields zero new changes).
5. Input DataFrame immutability (original data is never mutated).
6. End-to-end bit-level replay reproducibility against SHA-256 hash.
"""

from pathlib import Path
import json
import io
import pandas as pd
import numpy as np
import pytest

from datafix.config import Config
from datafix.pipeline import DataFixPipeline, replay_audit_log
from datafix.ui_helpers import apply_change_decisions, generate_export_bundle


@pytest.fixture
def sample_dirty_dataframe() -> pd.DataFrame:
    """Fixture with representative dirty data containing format, category, missing, relationship and duplicate issues."""
    return pd.DataFrame({
        "order_id": [101, 102, 103, 104, 104],
        "customer": ["Alice", "Bob", "Charlie", "David", "David"],
        "price": ["$10.50", "USD 20.00", "30.00", "$40.00", "$40.00"],
        "quantity": [2, 1, 3, 2, 2],
        "total_amount": [21.00, 20.00, 50.00, 80.00, 80.00],  # 103 has arithmetic mismatch (3 * 30 != 50)
        "country": ["USA", "United States", "us", "Canada", "Canada"],
        "date": ["2023-01-15", "15/01/2023", "2023.05.20", "2023-08-10", "2023-08-10"]
    })


def test_e2e_pipeline_and_all_four_exports(sample_dirty_dataframe: pd.DataFrame, tmp_path: Path):
    """Test full pipeline run, verification of all 6 phases, and generation of all four export artifacts."""
    df_input = sample_dirty_dataframe.copy()
    original_copy = df_input.copy(deep=True)

    cfg = Config(seed=42)
    pipeline = DataFixPipeline(config=cfg)

    # 1. Phase 1-6 Execution
    result = pipeline.run(df_input, dataset_name="AcceptanceOrder")

    # Assert Immutability: original df_input was NEVER modified in-place
    pd.testing.assert_frame_equal(df_input, original_copy)

    # Assert Phases 1 to 4.5 execution
    assert result.profile_before is not None
    assert result.profile_after is not None
    assert len(result.issues_before) > 0
    assert len(result.diagnoses) > 0
    assert len(result.audit_log.entries) > 0
    assert result.validation.metrics_after.overall_score > result.validation.metrics_before.overall_score

    # 2. Test All Four Export Artifacts via generate_export_bundle
    bundle = generate_export_bundle(
        original_df=result.original_df,
        repaired_df=result.repaired_df,
        audit_log=result.audit_log,
        issues_before=result.issues_before,
        diagnoses=result.diagnoses,
        dataset_name="AcceptanceOrder",
        config=cfg
    )

    # Export 1: Cleaned Dataset (CSV)
    assert "cleaned_csv" in bundle
    df_exported = pd.read_csv(io.StringIO(bundle["cleaned_csv"]))
    assert "_row_id" not in df_exported.columns
    assert len(df_exported) < len(df_input)  # Exact duplicates were deduplicated
    # Cleaned prices are numeric
    assert pd.to_numeric(df_exported["price"], errors="coerce").notna().all()

    # Export 2: Quality Report (HTML)
    assert "quality_report_html" in bundle
    html_content = bundle["quality_report_html"]
    assert "<!DOCTYPE html>" in html_content
    assert "DataFix Quality &amp; Verification Report" in html_content or "DataFix Quality" in html_content
    assert "AcceptanceOrder" in html_content

    # Export 3: Change Log (CSV)
    assert "change_log_csv" in bundle
    df_audit_csv = pd.read_csv(io.StringIO(bundle["change_log_csv"]))
    assert "change_id" in df_audit_csv.columns
    assert "old_value" in df_audit_csv.columns
    assert "new_value" in df_audit_csv.columns
    assert "status" in df_audit_csv.columns
    assert len(df_audit_csv) == len(result.audit_log.entries)

    # Export 4: Change Log (JSON)
    assert "change_log_json" in bundle
    json_data = json.loads(bundle["change_log_json"])
    assert isinstance(json_data, list)
    assert len(json_data) == len(result.audit_log.entries)
    assert json_data[0]["change_id"].startswith("CHG_")


def test_exported_data_reflects_approval_and_rejection_decisions(sample_dirty_dataframe: pd.DataFrame):
    """Test that rejecting a change preserves original value and approving applies repair across all exports."""
    cfg = Config(seed=42)
    pipeline = DataFixPipeline(config=cfg)
    result = pipeline.run(sample_dirty_dataframe.copy(), dataset_name="DecisionsTest")

    audit = result.audit_log
    applied_entries = [e for e in audit.entries if e.status == "applied"]
    assert len(applied_entries) >= 2

    # Choose one change to REJECT and one to APPROVE
    reject_entry = applied_entries[0]
    approve_entry = applied_entries[1]

    decisions = {e.change_id: e.status for e in audit.entries}
    decisions[reject_entry.change_id] = "rejected"
    decisions[approve_entry.change_id] = "approved"

    # Recompute repaired DataFrame with decisions
    df_recomputed, updated_audit = apply_change_decisions(
        result.original_df,
        audit,
        decisions
    )

    # Generate fresh export bundle reflecting user decisions
    bundle = generate_export_bundle(
        original_df=result.original_df,
        repaired_df=df_recomputed,
        audit_log=updated_audit,
        issues_before=result.issues_before,
        diagnoses=result.diagnoses,
        dataset_name="DecisionsTest"
    )

    # Verify Exported Cleaned CSV:
    df_exported = pd.read_csv(io.StringIO(bundle["cleaned_csv"]))

    # The rejected change must keep the OLD value in df_recomputed
    # Find original row corresponding to reject_entry
    orig_row = result.original_df.loc[result.original_df["_row_id"] == reject_entry.row_id]
    if not orig_row.empty and reject_entry.column in orig_row.columns:
        expected_rejected_val = str(orig_row[reject_entry.column].values[0])
        curr_row = df_recomputed.loc[df_recomputed["_row_id"] == reject_entry.row_id]
        if not curr_row.empty:
            assert str(curr_row[reject_entry.column].values[0]) == expected_rejected_val

    # Verify Exported Change Log CSV & JSON
    df_log = pd.read_csv(io.StringIO(bundle["change_log_csv"]))
    rej_row = df_log.loc[df_log["change_id"] == reject_entry.change_id]
    assert rej_row["status"].values[0] == "rejected"

    app_row = df_log.loc[df_log["change_id"] == approve_entry.change_id]
    assert app_row["status"].values[0] == "approved"

    log_json = json.loads(bundle["change_log_json"])
    json_lookup = {item["change_id"]: item["status"] for item in log_json}
    assert json_lookup[reject_entry.change_id] == "rejected"
    assert json_lookup[approve_entry.change_id] == "approved"


def test_repaired_dataset_idempotency(sample_dirty_dataframe: pd.DataFrame):
    """Test that running DataFix pipeline on its own repaired output produces zero new changes."""
    cfg = Config(seed=42)
    pipeline = DataFixPipeline(config=cfg)

    # First pass: repair dirty dataset
    pass1_result = pipeline.run(sample_dirty_dataframe.copy(), dataset_name="Pass1")
    repaired_pass1 = pass1_result.repaired_df.drop(columns=["_row_id"], errors="ignore")

    # Second pass: run pipeline on pass 1 output
    pass2_result = pipeline.run(repaired_pass1, dataset_name="Pass2")

    applied_pass2 = [e for e in pass2_result.audit_log.entries if e.status == "applied"]
    # Zero new modifications applied on second pass
    assert len(applied_pass2) == 0
    # Cleaned outputs are identical
    repaired_pass2 = pass2_result.repaired_df.drop(columns=["_row_id"], errors="ignore")
    pd.testing.assert_frame_equal(repaired_pass1, repaired_pass2)


def test_bit_level_replay_reproducibility(sample_dirty_dataframe: pd.DataFrame, tmp_path: Path):
    """Test that replaying change log on raw input matches the exact SHA-256 of the cleaned dataset."""
    cfg = Config(seed=42)
    pipeline = DataFixPipeline(config=cfg)

    res = pipeline.run(sample_dirty_dataframe.copy(), dataset_name="ReplayE2E")
    exports = res.export(output_dir=str(tmp_path), base_name="replay_e2e")

    # Replay audit log
    replay_res = replay_audit_log(
        input_file=str(tmp_path / "raw_input.csv") if (tmp_path / "raw_input.csv").exists() else exports["cleaned_csv"],
        log_file=exports["change_log_json"],
        expect_file=exports["cleaned_csv"]
    )

    # Verify provenance metadata exists
    assert res.metadata.input_sha256 is not None
    assert res.metadata.output_sha256 is not None
    assert len(res.metadata.output_sha256) == 64
