"""Tests for University Student Demo Dataset: Domain range validations & 7 issues detection."""

import pytest
import pandas as pd
from pathlib import Path

from datafix.detectors import run_all_detectors
from datafix.detectors.outliers import detect_outliers
from datafix.pipeline import DataFixPipeline


@pytest.fixture
def university_df():
    data_path = Path(__file__).parent.parent / "data" / "university_student_demo.csv"
    assert data_path.exists(), f"University dataset missing at {data_path}"
    return pd.read_csv(data_path)


def test_university_demo_all_7_issues_detected(university_df):
    """Verify that all 7 expected issues across 5 core categories are detected."""
    issues = run_all_detectors(university_df)
    assert len(issues) == 7, f"Expected exactly 7 issues, got {len(issues)}: {[i.issue_id for i in issues]}"

    issue_types = {i.issue_type for i in issues}
    assert "missing_value" in issue_types
    assert "duplicate_row" in issue_types
    assert "invalid_format" in issue_types
    assert "inconsistent_category" in issue_types
    assert "outlier" in issue_types

    # Map by issue_id
    issues_by_id = {i.issue_id: i for i in issues}

    # 1. Missing Email
    assert "MISS_Email" in issues_by_id
    assert issues_by_id["MISS_Email"].severity in ("high", "med")

    # 2. Missing Age
    assert "MISS_Age" in issues_by_id

    # 3. Duplicate row S006
    assert "DUP_EXACT" in issues_by_id
    assert issues_by_id["DUP_EXACT"].severity == "high"

    # 4. Invalid email format
    assert "FMT_EMAIL_Email" in issues_by_id
    assert "kabir.rao.univ.edu" in str(issues_by_id["FMT_EMAIL_Email"].sample_values)

    # 5. Inconsistent Department category
    assert "INCONSISTENT_Department" in issues_by_id

    # 6. GPA range violation
    assert "OUTLIER_UNI_GPA" in issues_by_id
    gpa_issue = issues_by_id["OUTLIER_UNI_GPA"]
    assert gpa_issue.columns == ["GPA"]
    assert gpa_issue.severity == "high"
    assert "0.0" in gpa_issue.description and "10.0" in gpa_issue.description
    assert gpa_issue.evidence["domain_bounds"] == {"min": 0.0, "max": 10.0}
    assert 10.8 in gpa_issue.evidence["invalid_values"]
    assert gpa_issue.evidence["violation_type"] == "domain_range_violation"

    # 7. Attendance_Percent range violation
    assert "OUTLIER_UNI_Attendance_Percent" in issues_by_id
    att_issue = issues_by_id["OUTLIER_UNI_Attendance_Percent"]
    assert att_issue.columns == ["Attendance_Percent"]
    assert att_issue.severity == "high"
    assert "0.0" in att_issue.description and "100.0" in att_issue.description
    assert att_issue.evidence["domain_bounds"] == {"min": 0.0, "max": 100.0}
    assert 105 in att_issue.evidence["invalid_values"] or 105.0 in att_issue.evidence["invalid_values"]
    assert att_issue.evidence["violation_type"] == "domain_range_violation"


def test_gpa_range_validation():
    """Verify GPA range bounds [0, 10] detect both upper and lower violations."""
    df = pd.DataFrame({
        "Student_ID": ["S1", "S2", "S3"],
        "GPA": [-0.5, 9.2, 11.0]
    })
    issues = detect_outliers(df)
    gpa_issues = [i for i in issues if "GPA" in i.columns]
    assert len(gpa_issues) == 1
    issue = gpa_issues[0]
    assert issue.severity == "high"
    assert issue.evidence["domain_bounds"] == {"min": 0.0, "max": 10.0}
    assert set(issue.evidence["invalid_values"]) == {-0.5, 11.0}


def test_attendance_range_validation():
    """Verify Attendance bounds [0, 100] detect violations."""
    df = pd.DataFrame({
        "Student_ID": ["S1", "S2", "S3"],
        "Attendance_Percent": [95, -2, 104]
    })
    issues = detect_outliers(df)
    att_issues = [i for i in issues if "Attendance_Percent" in i.columns]
    assert len(att_issues) == 1
    issue = att_issues[0]
    assert issue.severity == "high"
    assert issue.evidence["domain_bounds"] == {"min": 0.0, "max": 100.0}
    assert set(issue.evidence["invalid_values"]) == {-2, 104}


def test_university_demo_pipeline_e2e(university_df):
    """Verify end-to-end pipeline execution on university demo data."""
    pipe = DataFixPipeline()
    res = pipe.run(university_df, dataset_name="university_student_demo.csv")

    assert len(res.issues_before) == 7
    assert res.validation.metrics_after.overall_score > res.validation.metrics_before.overall_score

    # Check repairs applied
    applied_changes = [e for e in res.audit_log.entries if e.status == "applied"]
    gpa_repair = next((e for e in applied_changes if e.column == "GPA"), None)
    assert gpa_repair is not None
    assert gpa_repair.new_value == 10.0
    assert gpa_repair.confidence == 0.85

    att_repair = next((e for e in applied_changes if e.column == "Attendance_Percent"), None)
    assert att_repair is not None
    assert att_repair.new_value == 100
    assert att_repair.confidence == 0.85
