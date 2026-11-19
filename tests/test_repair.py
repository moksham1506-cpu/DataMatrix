"""Tests for datafix/repair.py: Safe repair strategies, immutability, and strict idempotency."""

import numpy as np
import pandas as pd
import pytest

from datafix.config import Config
from datafix.detectors import run_all_detectors
from datafix.repair import repair_dataset


def test_repair_immutability():
    """Verify that repair_dataset NEVER mutates the input dataframe."""
    df_orig = pd.DataFrame({
        "country": ["USA", "United States", "us"],
        "price": ["$10.50", "$20.00", "$30.00"]
    })
    df_copy = df_orig.copy(deep=True)

    issues = run_all_detectors(df_orig)
    df_repaired, audit = repair_dataset(df_orig, issues)

    # df_orig must remain identical to its initial copy
    pd.testing.assert_frame_equal(df_orig, df_copy)
    # df_repaired must have changes applied
    assert "United States" in df_repaired["country"].values
    assert df_repaired["price"].iloc[0] == 10.50


def test_repair_confidence_gating():
    """Verify that repairs >= 0.85 are applied, and < 0.85 remain suggested."""
    df = pd.DataFrame({
        "name": ["Alice Smith", "Alice Smiith"],  # fuzzy duplicate (confidence < 0.85)
        "price": ["$10.00", "$20.00"]             # numeric string parse (confidence >= 0.85)
    })
    cfg = Config(auto_apply_threshold=0.85)
    issues = run_all_detectors(df, config=cfg)
    df_repaired, audit = repair_dataset(df, issues, config=cfg)

    # Price should be auto-applied
    applied = audit.filter_by_status("applied")
    assert any(e.strategy == "parse_numeric_string" for e in applied)
    assert df_repaired["price"].iloc[0] == 10.0

    # Fuzzy duplicate should only be suggested
    suggested = audit.filter_by_status("suggested")
    assert any("fuzzy" in e.strategy for e in suggested)
    # The fuzzy duplicate row must NOT have been dropped
    assert len(df_repaired) == 2


def test_repair_disguised_nulls_and_categories():
    df = pd.DataFrame({
        "phone": ["+1-555-1234", "N/A", "null"],
        "country": ["USA", "U.S.A.", "United States"]
    })
    issues = run_all_detectors(df)
    df_repaired, audit = repair_dataset(df, issues)

    # Disguised nulls standardized to NaN
    assert pd.isna(df_repaired["phone"].iloc[1])
    assert pd.isna(df_repaired["phone"].iloc[2])

    # Categories mapped to canonical form
    assert (df_repaired["country"] == "United States").all()


def test_repair_arithmetic_relationship():
    df = pd.DataFrame({
        "quantity": [2, 5],
        "unit_price": [10.0, 20.0],
        "total_amount": [20.0, 50.0]  # second row has mismatch: 5 * 20 = 100 != 50
    })
    issues = run_all_detectors(df)
    df_repaired, audit = repair_dataset(df, issues)

    # Total amount recalculated
    assert df_repaired["total_amount"].iloc[1] == 100.0
    applied = audit.filter_by_status("applied")
    assert any(e.strategy == "recalculate_arithmetic_total" for e in applied)


def test_repair_idempotency():
    """CRITICAL TEST: Running detection and repair on the repaired output must produce zero new changes."""
    df_messy = pd.DataFrame({
        "order_date": ["2023-01-01", "15/01/2023", "2023.01.20"],
        "country": ["USA", "United States", "USA"],
        "price": ["$10.50", "$20.00", "$30.00"],
        "quantity": [1, 2, 3],
        "total_amount": [10.50, 40.0, 90.0]
    })
    cfg = Config()

    # Pass 1: detect and repair
    issues_pass1 = run_all_detectors(df_messy, config=cfg)
    df_repaired_1, audit_1 = repair_dataset(df_messy, issues_pass1, config=cfg)

    # Pass 2: detect and repair on df_repaired_1
    issues_pass2 = run_all_detectors(df_repaired_1, config=cfg)
    df_repaired_2, audit_2 = repair_dataset(df_repaired_1, issues_pass2, config=cfg)

    # Must produce ZERO applied changes in pass 2
    applied_pass2 = audit_2.filter_by_status("applied")
    assert len(applied_pass2) == 0, f"Pass 2 generated unexpected applied repairs: {applied_pass2}"
    
    # Dataframe in pass 2 must be identical to pass 1
    # Compare content columns (ignoring dtypes nuances like object vs float if values equal)
    for col in df_repaired_1.columns:
        if col != "_row_id":
            vals1 = df_repaired_1[col].tolist()
            vals2 = df_repaired_2[col].tolist()
            assert vals1 == vals2, f"Column '{col}' changed between pass 1 and pass 2!"
