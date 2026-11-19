"""DataFix Validation Engine: Evaluates data quality metrics, score improvement, and defect resolution."""

from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from datafix.config import Config
from datafix.models import Issue
from datafix.detectors import run_all_detectors


@dataclass
class QualityMetrics:
    """Individual dimension scores and composite Data Quality Score (0–100)."""
    completeness: float
    validity: float
    uniqueness: float
    consistency: float
    overall_score: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class ValidationResult:
    """Detailed comparison of dataset quality before and after repairs."""
    metrics_before: QualityMetrics
    metrics_after: QualityMetrics
    score_delta: float
    issues_before_count: int
    issues_resolved_count: int
    issues_remaining_count: int
    issues_newly_introduced_count: int
    resolved_issue_ids: List[str] = field(default_factory=list)
    remaining_issues: List[Issue] = field(default_factory=list)
    unresolved_reasons: Dict[str, str] = field(default_factory=dict)
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metrics_before": self.metrics_before.to_dict(),
            "metrics_after": self.metrics_after.to_dict(),
            "score_delta": round(self.score_delta, 2),
            "issues_before_count": self.issues_before_count,
            "issues_resolved_count": self.issues_resolved_count,
            "issues_remaining_count": self.issues_remaining_count,
            "issues_newly_introduced_count": self.issues_newly_introduced_count,
            "resolved_issue_ids": self.resolved_issue_ids,
            "remaining_issues": [i.to_dict() for i in self.remaining_issues],
            "unresolved_reasons": self.unresolved_reasons,
            "is_valid": self.is_valid,
        }


def compute_quality_metrics(
    df: pd.DataFrame,
    issues: List[Issue],
    config: Optional[Config] = None
) -> QualityMetrics:
    """Compute weighted completeness, validity, uniqueness, and consistency scores (0–100)."""
    cfg = config or Config()
    total_rows = len(df)
    content_cols = [c for c in df.columns if c != "_row_id"]
    n_cols = max(len(content_cols), 1)
    total_cells = max(total_rows * n_cols, 1)

    if total_rows == 0:
        return QualityMetrics(100.0, 100.0, 100.0, 100.0, 100.0)

    # 1. Completeness: percentage of non-null cells
    missing_cells = 0
    for i in issues:
        if i.issue_type == "missing_value":
            missing_cells += len(i.row_ids)
    completeness = max(0.0, min(100.0, 100.0 * (1.0 - (missing_cells / total_cells))))

    # 2. Validity: percentage of cells with valid formats and domain ranges
    invalid_cells = 0
    for i in issues:
        if i.issue_type in {"invalid_format", "outlier"}:
            invalid_cells += len(i.row_ids)
    validity = max(0.0, min(100.0, 100.0 * (1.0 - (invalid_cells / total_cells))))

    # 3. Uniqueness: percentage of non-duplicate rows
    duplicate_rows = 0
    for i in issues:
        if i.issue_type == "duplicate_row":
            duplicate_rows += len(i.row_ids)
    uniqueness = max(0.0, min(100.0, 100.0 * (1.0 - (duplicate_rows / total_rows))))

    # 4. Consistency: percentage of rows without categorical or relational discrepancies
    inconsistent_rows = 0
    for i in issues:
        if i.issue_type in {"inconsistent_category", "relationship_violation"}:
            inconsistent_rows += len(i.row_ids)
    consistency = max(0.0, min(100.0, 100.0 * (1.0 - (inconsistent_rows / total_rows))))

    # Composite weighted overall score
    overall = (
        (completeness * cfg.weight_completeness)
        + (validity * cfg.weight_validity)
        + (uniqueness * cfg.weight_uniqueness)
        + (consistency * cfg.weight_consistency)
    )

    return QualityMetrics(
        completeness=round(completeness, 2),
        validity=round(validity, 2),
        uniqueness=round(uniqueness, 2),
        consistency=round(consistency, 2),
        overall_score=round(overall, 2),
    )


def validate_repair(
    df_original: pd.DataFrame,
    df_repaired: pd.DataFrame,
    issues_before: List[Issue],
    ref_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None
) -> ValidationResult:
    """Validate repairs by re-running all detectors, computing scores, and verifying defect resolution."""
    cfg = config or Config()

    # Re-run all detectors on the repaired dataset
    issues_after = run_all_detectors(df_repaired, ref_dfs=ref_dfs, config=cfg)

    # Compute quality metrics before and after
    metrics_before = compute_quality_metrics(df_original, issues_before, config=cfg)
    metrics_after = compute_quality_metrics(df_repaired, issues_after, config=cfg)
    score_delta = round(metrics_after.overall_score - metrics_before.overall_score, 2)

    # Determine resolved vs remaining issues
    before_signatures: Dict[str, Issue] = {
        f"{i.issue_type}:{','.join(sorted(i.columns))}": i for i in issues_before
    }
    after_signatures: Dict[str, Issue] = {
        f"{i.issue_type}:{','.join(sorted(i.columns))}": i for i in issues_after
    }

    resolved_issue_ids: List[str] = []
    for sig, issue in before_signatures.items():
        if sig not in after_signatures:
            # Entire issue resolved
            resolved_issue_ids.append(issue.issue_id)
        else:
            # Check if row count decreased significantly
            after_issue = after_signatures[sig]
            if len(after_issue.row_ids) < len(issue.row_ids):
                resolved_count = len(issue.row_ids) - len(after_issue.row_ids)
                if len(after_issue.row_ids) == 0:
                    resolved_issue_ids.append(issue.issue_id)

    # Check for newly introduced issues
    newly_introduced = [
        issue for sig, issue in after_signatures.items()
        if sig not in before_signatures
    ]

    # Explain reasons for unresolved issues
    unresolved_reasons: Dict[str, str] = {}
    for issue in issues_after:
        if issue.issue_type == "missing_value":
            unresolved_reasons[issue.issue_id] = (
                "Missing values in identifier/contact fields cannot be imputed without user verification."
            )
        elif issue.issue_type == "duplicate_row":
            unresolved_reasons[issue.issue_id] = (
                "Fuzzy duplicate candidates preserved; requires manual approval to merge records."
            )
        elif issue.issue_type == "relationship_violation":
            v_type = issue.evidence.get("violation_type", "")
            if v_type == "orphan_foreign_key":
                unresolved_reasons[issue.issue_id] = (
                    "Orphan foreign keys cannot be automatically synthesized; flagged for DBA review."
                )
            elif v_type == "chronological_inversion":
                unresolved_reasons[issue.issue_id] = (
                    "Inverted dates require confirmation of which timestamp is authoritative."
                )
        elif issue.issue_type == "invalid_format":
            unresolved_reasons[issue.issue_id] = (
                "Malformed email/calendar strings require user correction."
            )
        elif issue.issue_type == "outlier":
            unresolved_reasons[issue.issue_id] = (
                "Extreme numeric values preserved to prevent accidental deletion of valid edge cases."
            )

    is_valid = len(newly_introduced) == 0 and score_delta >= 0.0

    return ValidationResult(
        metrics_before=metrics_before,
        metrics_after=metrics_after,
        score_delta=score_delta,
        issues_before_count=len(issues_before),
        issues_resolved_count=len(resolved_issue_ids),
        issues_remaining_count=len(issues_after),
        issues_newly_introduced_count=len(newly_introduced),
        resolved_issue_ids=resolved_issue_ids,
        remaining_issues=issues_after,
        unresolved_reasons=unresolved_reasons,
        is_valid=is_valid,
    )
