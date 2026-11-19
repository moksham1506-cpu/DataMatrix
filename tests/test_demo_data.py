"""Tests for data/generate_demo_data.py."""

import json
from pathlib import Path
import pandas as pd
from data.generate_demo_data import generate_clean_datasets, inject_errors, generate_demo_files


def test_generate_clean_datasets():
    df_cust, df_ord = generate_clean_datasets(n_customers=100, n_orders=100, seed=42)
    assert len(df_cust) == 100
    assert len(df_ord) == 100
    assert "customer_id" in df_cust.columns
    assert "country" in df_cust.columns
    assert "total_amount" in df_ord.columns
    assert df_cust["email"].isna().sum() == 0


def test_inject_errors_coverage():
    df_cust, df_ord = generate_clean_datasets(n_customers=200, n_orders=200, seed=42)
    cust_err, ord_err, injections = inject_errors(df_cust, df_ord, seed=42)
    
    assert len(injections) > 0
    issue_types = {inj["issue_type"] for inj in injections}
    expected_types = {
        "missing_value",
        "duplicate_row",
        "invalid_format",
        "inconsistent_category",
        "outlier",
        "relationship_violation"
    }
    assert expected_types.issubset(issue_types), f"Missing issue types: {expected_types - issue_types}"


def test_generate_demo_files(tmp_path: Path):
    res = generate_demo_files(output_dir=str(tmp_path), seed=42, n_rows=150)
    
    assert Path(res["customers_csv"]).exists()
    assert Path(res["orders_csv"]).exists()
    assert Path(res["demo_dataset_csv"]).exists()
    assert Path(res["ground_truth_json"]).exists()
    
    with open(res["ground_truth_json"], "r", encoding="utf-8") as f:
        gt = json.load(f)
    assert gt["total_injections"] > 0
    assert "injections_by_type" in gt


def test_determinism():
    df1_c, df1_o, inj1 = inject_errors(*generate_clean_datasets(100, 100, seed=42), seed=42)
    df2_c, df2_o, inj2 = inject_errors(*generate_clean_datasets(100, 100, seed=42), seed=42)
    
    assert len(inj1) == len(inj2)
    assert [x["corrupted_value"] for x in inj1] == [x["corrupted_value"] for x in inj2]
