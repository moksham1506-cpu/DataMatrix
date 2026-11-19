"""DataFix Robustness Tests: Edge-case datasets, missing columns, corrupt encodings, and empty files."""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from datafix.pipeline import DataFixPipeline
from datafix.ui_helpers import validate_and_load_file


DATA_DIR = Path(__file__).parent / "data"


def test_no_id_dataset():
    """Verify datasets without an ID column have _row_id inserted and process cleanly."""
    file_path = DATA_DIR / "no_id.csv"
    assert file_path.exists()

    df, err = validate_and_load_file(file_path)
    assert err is None
    assert df is not None
    assert "_row_id" in df.columns
    assert len(df) == 5

    pipeline = DataFixPipeline()
    res = pipeline.run(df, dataset_name="no_id_test")
    assert res.repaired_df is not None
    assert len(res.repaired_df) == 5
    assert res.validation.is_valid


def test_all_text_dataset():
    """Verify datasets with only string/text columns process without numeric detector crashes."""
    file_path = DATA_DIR / "all_text.csv"
    assert file_path.exists()

    df, err = validate_and_load_file(file_path)
    assert err is None
    assert df is not None
    assert len(df.columns) == 6  # _row_id + 5 text cols

    pipeline = DataFixPipeline()
    res = pipeline.run(df, dataset_name="all_text_test")
    assert res.repaired_df is not None
    assert res.validation.metrics_after.overall_score >= 90.0


def test_mixed_many_nulls_dataset():
    """Verify datasets with severe missingness (50%-100% nulls) do not crash any stage."""
    file_path = DATA_DIR / "mixed_many_nulls.csv"
    assert file_path.exists()

    df, err = validate_and_load_file(file_path)
    assert err is None
    assert df is not None

    pipeline = DataFixPipeline()
    res = pipeline.run(df, dataset_name="many_nulls_test")
    assert res.repaired_df is not None
    assert len(res.issues_before) > 0
    assert res.validation.metrics_after.completeness <= 60.0


def test_single_column_file_handling():
    """Verify single-column files are gracefully rejected by UI helper with clear warning."""
    file_path = DATA_DIR / "single_column.csv"
    assert file_path.exists()

    df, err = validate_and_load_file(file_path)
    assert df is None
    assert err is not None
    assert "single column" in err.lower()

    # Even if passed directly into the pipeline, the pipeline must not crash
    raw_df = pd.read_csv(file_path)
    pipeline = DataFixPipeline()
    res = pipeline.run(raw_df, dataset_name="single_col_pipeline")
    assert res.repaired_df is not None


def test_empty_file_handling():
    """Verify 0-byte empty files are rejected gracefully without raising unhandled exceptions."""
    file_path = DATA_DIR / "empty.csv"
    assert file_path.exists()

    df, err = validate_and_load_file(file_path)
    assert df is None
    assert err is not None
    assert "empty" in err.lower() or "no readable data" in err.lower()


def test_bad_encoding_fallback():
    """Verify non-UTF8 files (Latin-1/CP1252) decode smoothly via encoding fallback."""
    file_path = DATA_DIR / "bad_encoding.csv"
    assert file_path.exists()

    df, err = validate_and_load_file(file_path)
    assert err is None
    assert df is not None
    assert len(df) == 3
    # Check that accented characters are preserved
    assert any("François" in str(v) or "Müller" in str(v) for v in df["name"])

    pipeline = DataFixPipeline()
    res = pipeline.run(df, dataset_name="bad_encoding_test")
    assert res.repaired_df is not None
