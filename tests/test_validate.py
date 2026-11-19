"""Tests for datafix/validate.py and datafix/report.py."""

import pandas as pd
from datafix.config import Config
from datafix.detectors import run_all_detectors
from datafix.repair import repair_dataset
from datafix.validate import compute_quality_metrics, validate_repair
from datafix.report import generate_html_report
from datafix.diagnose import diagnose_root_causes


def test_quality_metrics_calculation():
    df = pd.DataFrame({
        "_row_id": [0, 1, 2, 3],
        "col1": [1, 2, 3, 4],
        "col2": ["A", "B", "C", "D"]
    })
    # Clean dataset with 0 issues
    metrics = compute_quality_metrics(df, issues=[])
    assert metrics.completeness == 100.0
    assert metrics.validity == 100.0
    assert metrics.uniqueness == 100.0
    assert metrics.consistency == 100.0
    assert metrics.overall_score == 100.0


def test_validate_repair_improvement():
    df = pd.DataFrame({
        "price": ["$10.50", "$20.00", "$30.00", "USD 40.00"],
        "country": ["USA", "United States", "USA", "us"]
    })
    issues_before = run_all_detectors(df)
    df_repaired, audit = repair_dataset(df, issues_before)
    validation = validate_repair(df, df_repaired, issues_before)

    assert validation.score_delta >= 0.0
    assert validation.issues_newly_introduced_count == 0
    assert validation.is_valid is True
    assert validation.issues_resolved_count > 0


def test_generate_html_report(tmp_path):
    df = pd.DataFrame({
        "country": ["USA", "United States"],
        "price": ["$10.00", "$20.00"]
    })
    issues = run_all_detectors(df)
    diagnoses = diagnose_root_causes(issues, df)
    df_repaired, audit = repair_dataset(df, issues)
    validation = validate_repair(df, df_repaired, issues)

    out_file = tmp_path / "test_report.html"
    html = generate_html_report(
        validation_result=validation,
        diagnoses=diagnoses,
        audit_log=audit,
        dataset_name="TestDataset",
        output_path=str(out_file)
    )

    assert "<!DOCTYPE html>" in html
    assert "DataFix Quality & Audit Report" in html
    assert "TestDataset" in html
    assert out_file.exists()


def test_collateral_damage_assessment():
    df_orig = pd.DataFrame({
        "_row_id": [0, 1, 2, 3],
        "user_id": [101, 102, 103, 104],
        "country": ["USA", "United States", "Canada", "Germany"],
        "price": ["$10.00", "$20.00", "30.00", "40.00"]
    })
    # Issues detected on rows 0 and 1
    issues = run_all_detectors(df_orig)
    df_rep, audit = repair_dataset(df_orig, issues)
    validation = validate_repair(df_orig, df_rep, issues)

    cd = validation.collateral_damage
    assert cd is not None
    # 1. (a) % of originally valid cells left unchanged should be 100%
    assert cd.valid_cells_unchanged_pct == 100.0
    # 2. (b) Changed cells in clean rows (rows 2 and 3) should be 0
    assert cd.changed_cells_in_clean_rows == 0
    # 3. (c) Primary key integrity before vs after preserved
    assert cd.pk_fk_integrity["integrity_preserved"] is True
    assert cd.pk_fk_integrity["primary_keys"]["user_id"]["preserved"] is True
    # 4. (d) Row count before vs after
    assert cd.row_count_before == 4
    assert cd.row_count_after == 4
    assert cd.rows_removed == 0
