"""Tests for DataFix detectors suite."""

import numpy as np
import pandas as pd
import pytest

from datafix.config import Config
from datafix.detectors.missing import detect_missing_values
from datafix.detectors.duplicates import detect_duplicates
from datafix.detectors.formats import detect_format_issues
from datafix.detectors.consistency import detect_inconsistencies
from datafix.detectors.outliers import detect_outliers
from datafix.detectors.relationships import detect_relationship_violations
from datafix.detectors import run_all_detectors


def test_missing_detector():
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "David"],
        "email": ["alice@ex.com", None, "N/A", "david@ex.com"],
        "phone": ["+1555", "-", "??", ""]
    })
    issues = detect_missing_values(df)
    assert len(issues) == 2
    
    email_issue = next(i for i in issues if "email" in i.columns)
    assert email_issue.issue_type == "missing_value"
    assert email_issue.evidence["standard_null_count"] == 1
    assert email_issue.evidence["disguised_null_count"] == 1
    assert len(email_issue.row_ids) == 2


def test_duplicates_detector():
    df = pd.DataFrame({
        "name": ["Alice Smith", "Alice Smith", "Alice Smiith", "Bob Johnson"],
        "city": ["New York", "New York", "New York", "Boston"]
    })
    issues = detect_duplicates(df)
    issue_types = [i.issue_type for i in issues]
    assert "duplicate_row" in issue_types
    
    # Exact duplicate
    exact_issue = next((i for i in issues if i.evidence.get("duplicate_type") == "exact"), None)
    assert exact_issue is not None
    assert 1 in exact_issue.row_ids


def test_formats_detector():
    df = pd.DataFrame({
        "order_date": ["2023-01-01", "2023-01-02", "15/01/2023", "2023.01.20", "2023-02-31"],
        "email": ["valid@test.com", "not-an-email", "user@@broken.com", "valid2@test.com", "valid3@test.com"],
        "amount": ["$100.50", "USD 250", "$300.00", "400.00", "$50.00"]
    })
    issues = detect_format_issues(df)
    assert len(issues) >= 3
    cols_flagged = {i.columns[0] for i in issues}
    assert "order_date" in cols_flagged
    assert "email" in cols_flagged
    assert "amount" in cols_flagged


def test_consistency_detector():
    df = pd.DataFrame({
        "country": ["United States", "USA", "United States", "us", "Canada", "Canada"],
        "tier": ["Gold", "gold", "  Gold  ", "Silver", "Silver", "Silver"]
    })
    issues = detect_inconsistencies(df)
    assert len(issues) == 2
    cols_flagged = {i.columns[0] for i in issues}
    assert "country" in cols_flagged
    assert "tier" in cols_flagged


def test_outliers_detector():
    # Construct a dataset with normal values and a clear extreme outlier
    ages = [25, 28, 30, 32, 35, 40, 42, 45, 50, 52, 230, -5]
    prices = [10.0, 12.0, 15.0, 14.0, 11.0, 13.0, 16.0, 12.5, 14.2, 13.8, 99999.0, 11.5]
    df = pd.DataFrame({"age": ages, "unit_price": prices})
    
    issues = detect_outliers(df)
    assert len(issues) >= 2
    cols_flagged = [i.columns[0] for i in issues if len(i.columns) == 1]
    assert "age" in cols_flagged or "unit_price" in cols_flagged


def test_relationships_detector():
    df_customers = pd.DataFrame({
        "customer_id": [101, 102, 103],
        "name": ["A", "B", "C"]
    })
    df_orders = pd.DataFrame({
        "order_id": [1, 2, 3, 4],
        "customer_id": [101, 102, 999, 101],
        "order_date": ["2023-05-10", "2023-05-15", "2023-06-01", "2023-06-10"],
        "delivery_date": ["2023-05-12", "2023-05-01", "2023-06-05", "2023-06-15"],  # row 1 inverted
        "quantity": [2, 3, 1, 4],
        "unit_price": [10.0, 20.0, 50.0, 25.0],
        "total_amount": [20.0, 60.0, 150.0, 100.0]  # row 2 mismatch: 1 * 50 != 150
    })
    
    issues = detect_relationship_violations(df_orders, ref_dfs={"customers": df_customers})
    assert len(issues) == 3
    violation_types = {i.evidence.get("violation_type") for i in issues}
    assert "orphan_foreign_key" in violation_types
    assert "chronological_inversion" in violation_types
    assert "arithmetic_mismatch" in violation_types


def test_run_all_detectors():
    df = pd.DataFrame({
        "name": ["Alice", "Alice", "Bob", "N/A"],
        "email": ["alice@ex.com", "alice@ex.com", "bad-email", "bob@ex.com"],
        "country": ["USA", "USA", "United States", "Canada"]
    })
    all_issues = run_all_detectors(df)
    assert len(all_issues) >= 3
    found_types = {i.issue_type for i in all_issues}
    assert "missing_value" in found_types
    assert "duplicate_row" in found_types
    assert "invalid_format" in found_types
