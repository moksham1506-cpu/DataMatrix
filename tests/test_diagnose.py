"""Tests for datafix/diagnose.py."""

from datafix.diagnose import diagnose_root_causes
from datafix.detectors import run_all_detectors
import pandas as pd


def test_diagnose_root_causes():
    df = pd.DataFrame({
        "signup_date": ["2023-01-01", "15/01/2023", "2023.01.20", "2023-01-05"],
        "price": ["$10.50", "USD 20.00", "$15.00", "$30.00"],
        "phone": ["+1-555-1234", "N/A", "null", "+1-555-5678"],
        "country": ["USA", "United States", "us", "Canada"]
    })
    issues = run_all_detectors(df)
    diagnoses = diagnose_root_causes(issues, df)

    assert len(diagnoses) >= 3
    labels = [d.root_cause_label for d in diagnoses]
    
    # Should diagnose disguised nulls, date formats, and currency strings
    has_null_diag = any("Null" in l for l in labels)
    has_date_diag = any("Date" in l for l in labels)
    has_curr_diag = any("Currency" in l for l in labels)

    assert has_null_diag
    assert has_date_diag
    assert has_curr_diag
