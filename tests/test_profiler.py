"""Tests for datafix/profiler.py."""

import pandas as pd
import numpy as np
from datafix.profiler import ensure_row_id, infer_column_type, profile_column, profile_dataset
from datafix.config import Config


def test_ensure_row_id_immutability():
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    df_with_id = ensure_row_id(df)
    
    assert "_row_id" not in df.columns
    assert "_row_id" in df_with_id.columns
    assert list(df_with_id["_row_id"]) == [0, 1]
    
    # If already has _row_id, returns copy without adding duplicate
    df_again = ensure_row_id(df_with_id)
    assert list(df_again["_row_id"]) == [0, 1]


def test_infer_column_types():
    disguised = set(Config().disguised_nulls)
    
    # Numeric
    s_num = pd.Series([10.5, 20.2, 30.1, np.nan], name="price")
    assert infer_column_type(s_num, disguised) == "numeric"
    
    # Formatted numeric string
    s_num_str = pd.Series(["$10.50", " $20.00 ", "$30.25", "$15.00"], name="revenue")
    assert infer_column_type(s_num_str, disguised) == "numeric"
    
    # Datetime
    s_date = pd.Series(["2023-01-01", "2023-01-02", "2023-01-03", "N/A"], name="created_at")
    assert infer_column_type(s_date, disguised) == "datetime"
    
    # Email
    s_email = pd.Series(["user1@example.com", "user2@test.org", "invalid", "user3@domain.co"], name="email")
    assert infer_column_type(s_email, disguised) == "email"
    
    # Phone
    s_phone = pd.Series(["+1-555-123-4567", "(555) 987-6543", "+44 20 7946 0958"], name="phone_number")
    assert infer_column_type(s_phone, disguised) == "phone"
    
    # Boolean
    s_bool = pd.Series(["true", "false", "true", "True"], name="is_active")
    assert infer_column_type(s_bool, disguised) == "boolean"
    
    # Identifier
    s_id = pd.Series([1001, 1002, 1003, 1004], name="customer_id")
    assert infer_column_type(s_id, disguised) == "identifier"
    
    # Category
    s_cat = pd.Series(["Gold", "Silver", "Bronze", "Gold", "Silver"], name="tier")
    assert infer_column_type(s_cat, disguised) == "category"


def test_profile_dataset():
    df = pd.DataFrame({
        "id": [1, 2, 3, 3],
        "name": ["Alice", "Bob", "Charlie", "Charlie"],
        "score": [95.0, np.nan, 88.5, 88.5],
        "status": ["active", "n/a", "active", "active"]
    })
    
    profile = profile_dataset(df)
    assert profile.row_count == 4
    assert profile.column_count == 4
    assert profile.duplicate_rows_count == 1
    
    status_prof = profile.column_profiles["status"]
    assert status_prof.disguised_null_count == 1
    assert status_prof.null_count == 0
    assert status_prof.null_ratio == 0.25
    
    score_prof = profile.column_profiles["score"]
    assert score_prof.inferred_type == "numeric"
    assert "mean" in score_prof.stats
