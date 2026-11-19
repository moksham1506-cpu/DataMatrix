"""Detector for standard and disguised missing values."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from datafix.config import Config
from datafix.models import Issue
from datafix.profiler import ensure_row_id


def detect_missing_values(df: pd.DataFrame, config: Optional[Config] = None) -> List[Issue]:
    """Detect standard NaNs, empty strings, and disguised missing values in a DataFrame."""
    cfg = config or Config()
    df_with_id = ensure_row_id(df)
    disguised_set: Set[str] = set(token.lower() for token in cfg.disguised_nulls)
    
    issues: List[Issue] = []
    total_rows = len(df_with_id)
    if total_rows == 0:
        return issues

    content_cols = [c for c in df_with_id.columns if c != "_row_id"]

    for col in content_cols:
        series = df_with_id[col]
        # 1. Standard nulls (NaN, None)
        std_null_mask = series.isna()

        # 2. Empty strings and disguised nulls (for object / string dtypes)
        disguised_mask = pd.Series(False, index=df_with_id.index)
        disguised_token_counts: Dict[str, int] = {}

        if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
            # Check empty strings
            str_series = series.dropna().astype(str).str.strip()
            empty_mask = str_series == ""
            
            # Check disguised null tokens
            str_lower = str_series.str.lower()
            token_mask = str_lower.isin(disguised_set)
            
            # Map back to full index
            full_disguised_mask = pd.Series(False, index=df_with_id.index)
            valid_indices = str_series.index
            full_disguised_mask.loc[valid_indices[empty_mask | token_mask]] = True
            disguised_mask = full_disguised_mask

            if token_mask.any():
                matched_tokens = str_series[token_mask].str.lower().value_counts().to_dict()
                disguised_token_counts = {str(k): int(v) for k, v in matched_tokens.items()}
            if empty_mask.any():
                disguised_token_counts["<empty_string>"] = int(empty_mask.sum())

        # Combine standard and disguised nulls
        combined_mask = std_null_mask | disguised_mask
        missing_count = int(combined_mask.sum())

        if missing_count > 0:
            missing_row_ids = df_with_id.loc[combined_mask, "_row_id"].tolist()
            missing_ratio = float(missing_count / total_rows)
            
            # Extract sample values (up to 5)
            samples = series.loc[combined_mask].head(5).tolist()

            # Severity determination
            col_lower = col.lower()
            is_key_col = col_lower.endswith(("_id", "id", "_key", "key")) or col_lower in {"email", "phone", "name"}
            if is_key_col or missing_ratio >= 0.20:
                severity = "high"
            elif missing_ratio >= 0.05:
                severity = "med"
            else:
                severity = "low"

            disguised_count = int(disguised_mask.sum())
            std_count = int(std_null_mask.sum())
            
            description = (
                f"Column '{col}' has {missing_count} missing values ({missing_ratio:.1%}): "
                f"{std_count} standard nulls and {disguised_count} disguised nulls."
            )

            evidence = {
                "total_missing": missing_count,
                "standard_null_count": std_count,
                "disguised_null_count": disguised_count,
                "missing_ratio": round(missing_ratio, 4),
                "disguised_token_counts": disguised_token_counts
            }

            issue = Issue(
                issue_id=f"MISS_{col}",
                issue_type="missing_value",
                columns=[col],
                row_ids=missing_row_ids,
                sample_values=samples,
                severity=severity,
                description=description,
                detector="missing_detector",
                evidence=evidence
            )
            issues.append(issue)

    return issues
