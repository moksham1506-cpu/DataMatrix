"""DataFix Detectors: Modular suite for detecting data quality issues."""

from __future__ import annotations
from typing import Dict, List, Optional
import pandas as pd

from datafix.config import Config
from datafix.models import Issue
from datafix.detectors.missing import detect_missing_values
from datafix.detectors.duplicates import detect_duplicates
from datafix.detectors.formats import detect_format_issues
from datafix.detectors.consistency import detect_inconsistencies
from datafix.detectors.outliers import detect_outliers
from datafix.detectors.relationships import detect_relationship_violations


__all__ = [
    "detect_missing_values",
    "detect_duplicates",
    "detect_format_issues",
    "detect_inconsistencies",
    "detect_outliers",
    "detect_relationship_violations",
    "run_all_detectors",
]


def run_all_detectors(
    df: pd.DataFrame,
    ref_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None
) -> List[Issue]:
    """Execute all data quality detectors and return an aggregated list of Issues."""
    cfg = config or Config()
    all_issues: List[Issue] = []

    # 1. Missing values (standard + disguised)
    all_issues.extend(detect_missing_values(df, config=cfg))

    # 2. Exact and fuzzy duplicates
    all_issues.extend(detect_duplicates(df, config=cfg))

    # 3. Format errors (dates, emails, phones, numeric-as-text)
    all_issues.extend(detect_format_issues(df, config=cfg))

    # 4. Consistency errors (case, whitespace, synonyms, fuzzy categories)
    all_issues.extend(detect_inconsistencies(df, config=cfg))

    # 5. Outliers (univariate IQR/Z-score/domain + multivariate Isolation Forest)
    all_issues.extend(detect_outliers(df, config=cfg))

    # 6. Relational violations (foreign key orphans, date inversions, arithmetic mismatches)
    all_issues.extend(detect_relationship_violations(df, ref_dfs=ref_dfs, config=cfg))

    return all_issues
