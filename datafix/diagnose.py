"""DataFix Diagnostic Engine: Groups detected issues into root-cause patterns and generates explanations."""

from __future__ import annotations
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
import pandas as pd

from datafix.config import Config
from datafix.models import Issue


@dataclass
class Diagnosis:
    """Represents an inferred root cause explaining one or more related data quality issues."""
    diagnosis_id: str
    root_cause_label: str
    affected_columns: List[str]
    issue_ids: List[str]
    affected_row_count: int
    confidence: float
    evidence: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Diagnosis:
        return cls(**data)


def diagnose_root_causes(
    issues: List[Issue],
    df: Optional[pd.DataFrame] = None,
    config: Optional[Config] = None
) -> List[Diagnosis]:
    """Cluster quality issues by statistical signature into root-cause diagnostic groups."""
    cfg = config or Config()
    diagnoses: List[Diagnosis] = []

    if not issues:
        return diagnoses

    # Group issues by issue_type and pattern
    by_type: Dict[str, List[Issue]] = {}
    for issue in issues:
        by_type.setdefault(issue.issue_type, []).append(issue)

    # 1. DIAGNOSE: Disguised Null Sentinel Tokens
    if "missing_value" in by_type:
        disguised_issues = [
            i for i in by_type["missing_value"]
            if i.evidence.get("disguised_null_count", 0) > 0
        ]
        if disguised_issues:
            cols = [c for i in disguised_issues for c in i.columns]
            all_row_ids = set(r for i in disguised_issues for r in i.row_ids)
            total_disguised = sum(i.evidence.get("disguised_null_count", 0) for i in disguised_issues)
            tokens = {}
            for i in disguised_issues:
                tokens.update(i.evidence.get("disguised_token_counts", {}))

            diagnoses.append(Diagnosis(
                diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                root_cause_label="Disguised Null Sentinel String Tokens",
                affected_columns=cols,
                issue_ids=[i.issue_id for i in disguised_issues],
                affected_row_count=len(all_row_ids),
                confidence=0.96,
                evidence={"total_disguised": total_disguised, "tokens": tokens},
                explanation=(
                    f"Found {total_disguised} disguised null placeholders across columns {cols}. "
                    f"Upstream exports or form processors serialized missing values as placeholder string tokens "
                    f"({list(tokens.keys())[:4]}) rather than native null values."
                ),
                recommended_action="Convert placeholder string tokens to native null (NaN) types for consistent missing value handling."
            ))

    # 2. DIAGNOSE: Multi-Source Disparate Date Formats
    if "invalid_format" in by_type:
        date_format_issues = [
            i for i in by_type["invalid_format"]
            if "DATE" in i.issue_id or "dominant_format" in i.evidence
        ]
        if date_format_issues:
            for issue in date_format_issues:
                dom_fmt = issue.evidence.get("dominant_format", "YYYY-MM-DD")
                breakdown = issue.evidence.get("format_breakdown", {})
                diagnoses.append(Diagnosis(
                    diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                    root_cause_label="Disparate Date Format Conventions from Multiple Upstream Feeds",
                    affected_columns=issue.columns,
                    issue_ids=[issue.issue_id],
                    affected_row_count=len(issue.row_ids),
                    confidence=0.94,
                    evidence={"dominant_format": dom_fmt, "format_breakdown": breakdown},
                    explanation=(
                        f"Column '{issue.columns[0]}' exhibits a mixture of date formatting conventions "
                        f"({[k for k, v in breakdown.items() if v > 0]}). This typically occurs when records "
                        f"are ingested from differing regional locales (e.g. US MM/DD vs International DD/MM) "
                        f"or unstandardized legacy systems."
                    ),
                    recommended_action=f"Normalize all date representations into ISO-8601 standard ({dom_fmt})."
                ))

    # 3. DIAGNOSE: Formatted Currency and Thousand Separators in Numeric Fields
    if "invalid_format" in by_type:
        num_issues = [
            i for i in by_type["invalid_format"]
            if i.evidence.get("format_type") == "numeric_as_text"
        ]
        if num_issues:
            cols = [c for i in num_issues for c in i.columns]
            all_row_ids = set(r for i in num_issues for r in i.row_ids)
            diagnoses.append(Diagnosis(
                diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                root_cause_label="Formatted Currency Strings in Numeric Quantitative Metrics",
                affected_columns=cols,
                issue_ids=[i.issue_id for i in num_issues],
                affected_row_count=len(all_row_ids),
                confidence=0.98,
                evidence={"dirty_columns": cols},
                explanation=(
                    f"Quantitative columns {cols} contain string formatting (currency symbols '$', 'USD' "
                    f"and commas). The data was exported from a spreadsheet or financial UI without unformatting."
                ),
                recommended_action="Strip currency glyphs and grouping commas, casting column to native float64."
            ))

    # 4. DIAGNOSE: Casing and Whitespace Data Entry Inconsistencies
    if "inconsistent_category" in by_type:
        cat_issues = by_type["inconsistent_category"]
        for issue in cat_issues:
            clusters = issue.evidence.get("clusters", [])
            has_casing = any(c.get("type") == "casing_whitespace" for c in clusters)
            has_synonym = any(c.get("type") == "synonym_abbreviation" for c in clusters)
            has_fuzzy = any(c.get("type") == "fuzzy_variant" for c in clusters)

            label = "Unconstrained Categorical Text Entry Discrepancies"
            if has_synonym:
                label = "Abbreviation and Master Data Taxonomy Inconsistencies"
            elif has_casing:
                label = "Case Sensitivity and Whitespace Input Variations"

            diagnoses.append(Diagnosis(
                diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                root_cause_label=label,
                affected_columns=issue.columns,
                issue_ids=[issue.issue_id],
                affected_row_count=len(issue.row_ids),
                confidence=0.92,
                evidence={"cluster_count": len(clusters), "sample_clusters": clusters[:3]},
                explanation=(
                    f"Column '{issue.columns[0]}' contains variant representations of identical canonical concepts. "
                    f"Root cause stems from free-text form inputs or decentralized data entry without validation dropdowns."
                ),
                recommended_action="Map categorical variants to their high-frequency canonical representation."
            ))

    # 5. DIAGNOSE: Duplicate Ingestion / Pipeline Replays
    if "duplicate_row" in by_type:
        dup_issues = by_type["duplicate_row"]
        exact_dups = [i for i in dup_issues if i.evidence.get("duplicate_type") == "exact"]
        if exact_dups:
            total_dup_rows = sum(len(i.row_ids) for i in exact_dups)
            diagnoses.append(Diagnosis(
                diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                root_cause_label="Pipeline Replay or Redundant Batch File Re-ingestion",
                affected_columns=["ALL"],
                issue_ids=[i.issue_id for i in exact_dups],
                affected_row_count=total_dup_rows,
                confidence=0.97,
                evidence={"exact_duplicate_rows": total_dup_rows},
                explanation=(
                    f"Identified {total_dup_rows} exact duplicate rows with identical values across all content fields. "
                    f"This typically indicates a batch ETL job re-run, webhook duplicate event emission, or repeated file ingestion."
                ),
                recommended_action="Deduplicate by retaining the primary canonical instance while preserving relational keys."
            ))

    # 6. DIAGNOSE: Relational Integrity & Workflow Inversions
    if "relationship_violation" in by_type:
        rel_issues = by_type["relationship_violation"]
        for issue in rel_issues:
            v_type = issue.evidence.get("violation_type")
            if v_type == "orphan_foreign_key":
                parent_table = issue.evidence.get("parent_table", "parent")
                diagnoses.append(Diagnosis(
                    diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                    root_cause_label=f"Referential Integrity Breach: Missing Parent Records in '{parent_table}'",
                    affected_columns=issue.columns,
                    issue_ids=[issue.issue_id],
                    affected_row_count=len(issue.row_ids),
                    confidence=0.95,
                    evidence=issue.evidence,
                    explanation=(
                        f"Found {len(issue.row_ids)} child records referencing nonexistent keys in table '{parent_table}'. "
                        f"Cause is likely out-of-order data pipeline synchronization, partial table archiving, or hard deletes on the parent."
                    ),
                    recommended_action="Flag orphan records for review; do not delete or invent foreign keys automatically."
                ))
            elif v_type == "chronological_inversion":
                diagnoses.append(Diagnosis(
                    diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                    root_cause_label="Workflow Sequence Inversion (Delivery Precedes Order)",
                    affected_columns=issue.columns,
                    issue_ids=[issue.issue_id],
                    affected_row_count=len(issue.row_ids),
                    confidence=0.91,
                    evidence=issue.evidence,
                    explanation=(
                        f"Columns {issue.columns} exhibit inverted chronological timestamps. "
                        f"Cause is manual retrospective data entry or timezone conversion errors between client and server."
                    ),
                    recommended_action="Flag chronological inversions as suggestions for manual confirmation."
                ))
            elif v_type == "arithmetic_mismatch":
                diagnoses.append(Diagnosis(
                    diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                    root_cause_label="Computed Metric Arithmetic Inconsistency",
                    affected_columns=issue.columns,
                    issue_ids=[issue.issue_id],
                    affected_row_count=len(issue.row_ids),
                    confidence=0.95,
                    evidence=issue.evidence,
                    explanation=(
                        f"Total column does not equal the calculated product of quantity and unit price in {len(issue.row_ids)} rows. "
                        f"Cause is an unrecorded discount/tax adjustment or downstream manual row updates."
                    ),
                    recommended_action="Recalculate total amount from primary quantity and unit price factors."
                ))

    # 7. DIAGNOSE: Physical Domain Outliers
    if "outlier" in by_type:
        outlier_issues = [i for i in by_type["outlier"] if i.evidence.get("domain_bounds")]
        for issue in outlier_issues:
            diagnoses.append(Diagnosis(
                diagnosis_id=f"DIAG_{uuid.uuid4().hex[:8]}",
                root_cause_label="Physical / Domain Range Anomaly",
                affected_columns=issue.columns,
                issue_ids=[issue.issue_id],
                affected_row_count=len(issue.row_ids),
                confidence=0.90,
                evidence=issue.evidence,
                explanation=(
                    f"Column '{issue.columns[0]}' contains values violating physical domain boundaries "
                    f"(bounds: {issue.evidence.get('domain_bounds')}). Cause is likely negative sentinel codes (-1, -99) "
                    f"or transcription typos."
                ),
                recommended_action="Clamp or flag impossible values outside domain constraints."
            ))

    return diagnoses
