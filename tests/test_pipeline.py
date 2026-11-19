"""Tests for datafix/pipeline.py and datafix/cli.py."""

from pathlib import Path
import pandas as pd
import pytest

from datafix.config import Config
from datafix.pipeline import DataFixPipeline, run_pipeline


def test_pipeline_end_to_end(tmp_path: Path):
    df = pd.DataFrame({
        "order_date": ["2023-01-01", "15/01/2023", "2023.01.20"],
        "country": ["USA", "United States", "us"],
        "total": ["$50.00", "$100.00", "$150.00"],
        "phone": ["+1-555-1234", "N/A", "+1-555-9876"]
    })
    
    result = run_pipeline(df, dataset_name="TestOrders")
    
    # Verify complete pipeline result artifacts
    assert result.original_df is not None
    assert result.repaired_df is not None
    assert len(result.issues_before) > 0
    assert len(result.diagnoses) > 0
    assert len(result.audit_log) > 0
    assert result.validation.score_delta >= 0.0
    assert result.validation.issues_newly_introduced_count == 0

    # Test export functionality
    exports = result.export(output_dir=str(tmp_path), base_name="orders_test")
    assert Path(exports["cleaned_csv"]).exists()
    assert Path(exports["change_log_csv"]).exists()
    assert Path(exports["change_log_json"]).exists()
    assert Path(exports["quality_report_html"]).exists()

    # Verify cleaned CSV does not have internal _row_id exposed
    df_cleaned = pd.read_csv(exports["cleaned_csv"])
    assert "_row_id" not in df_cleaned.columns


def test_cli_execution(tmp_path: Path):
    # Create sample CSV
    csv_file = tmp_path / "sample.csv"
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Alice"],
        "amount": ["$10.00", "$20.00", "$10.00"]
    })
    df.to_csv(csv_file, index=False)

    out_dir = tmp_path / "cli_outputs"

    from datafix.cli import main
    import sys

    test_args = ["datafix.cli", str(csv_file), "--out", str(out_dir), "--threshold", "0.80", "--quiet"]
    sys.argv = test_args
    try:
        main()
    except SystemExit as e:
        assert e.code in {0, None}

    assert (out_dir / "cleaned_sample.csv").exists()
    assert (out_dir / "quality_report_sample.html").exists()
    assert (out_dir / "change_log_sample.csv").exists()
