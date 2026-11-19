"""Detector for univariate (IQR, Z-score, domain bounds) and multivariate (IsolationForest) outliers."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from datafix.config import Config
from datafix.models import Issue
from datafix.profiler import ensure_row_id


DOMAIN_CONSTRAINTS = {
    "age": {"min": 0, "max": 130},
    "quantity": {"min": 1, "max": None},
    "unit_price": {"min": 0.0, "max": None},
    "price": {"min": 0.0, "max": None},
    "total_amount": {"min": 0.0, "max": None},
    "amount": {"min": 0.0, "max": None},
}


def detect_outliers(df: pd.DataFrame, config: Optional[Config] = None) -> List[Issue]:
    """Detect statistical and domain outliers across univariate and multivariate methods."""
    cfg = config or Config()
    df_with_id = ensure_row_id(df)
    issues: List[Issue] = []

    total_rows = len(df_with_id)
    if total_rows < 10:
        return issues

    content_cols = [c for c in df_with_id.columns if c != "_row_id"]

    # 1. Identify numeric columns
    numeric_cols: List[str] = []
    clean_numeric_df = pd.DataFrame(index=df_with_id.index)

    for col in content_cols:
        series = df_with_id[col]
        # Ignore identifier / ID columns
        if col.lower().endswith(("_id", "id", "_key", "key")):
            continue
        
        # Try numeric conversion
        num_s = pd.to_numeric(
            series.astype(str).str.replace(r"[\$,€£¥\s]", "", regex=True),
            errors="coerce"
        )
        # If >= 75% non-null numeric
        if num_s.notna().sum() / total_rows >= 0.75:
            numeric_cols.append(col)
            clean_numeric_df[col] = num_s

    # =========================================================================
    # 2. UNIVARIATE OUTLIER DETECTION
    # =========================================================================
    for col in numeric_cols:
        s = clean_numeric_df[col].dropna()
        if len(s) < 10:
            continue

        col_lower = col.lower()
        q25 = float(s.quantile(0.25))
        q75 = float(s.quantile(0.75))
        iqr = q75 - q25
        mean = float(s.mean())
        std = float(s.std()) if len(s) > 1 else 0.0

        lower_bound = q25 - (cfg.outlier_iqr_multiplier * iqr)
        upper_bound = q75 + (cfg.outlier_iqr_multiplier * iqr)

        # Domain boundary check
        domain_min = None
        domain_max = None
        for k, bounds in DOMAIN_CONSTRAINTS.items():
            if k in col_lower:
                domain_min = bounds["min"]
                domain_max = bounds["max"]
                break

        # Masks
        iqr_outliers = (clean_numeric_df[col] < lower_bound) | (clean_numeric_df[col] > upper_bound)
        
        z_outliers = pd.Series(False, index=clean_numeric_df.index)
        if std > 0:
            z_scores = (clean_numeric_df[col] - mean).abs() / std
            z_outliers = z_scores > cfg.outlier_zscore_threshold

        domain_outliers = pd.Series(False, index=clean_numeric_df.index)
        if domain_min is not None:
            domain_outliers = domain_outliers | (clean_numeric_df[col] < domain_min)
        if domain_max is not None:
            domain_outliers = domain_outliers | (clean_numeric_df[col] > domain_max)

        # Combined univariate outliers (must violate domain boundary OR (IQR and Z-score))
        combined_outlier_mask = domain_outliers | (iqr_outliers & z_outliers)
        outlier_count = int(combined_outlier_mask.sum())

        if outlier_count > 0:
            outlier_row_ids = df_with_id.loc[combined_outlier_mask, "_row_id"].tolist()
            sample_vals = df_with_id.loc[combined_outlier_mask, col].head(5).tolist()

            has_domain_violation = bool(domain_outliers.any())
            severity = "high" if has_domain_violation or outlier_count > 20 else "med"

            issues.append(Issue(
                issue_id=f"OUTLIER_UNI_{col}",
                issue_type="outlier",
                columns=[col],
                row_ids=outlier_row_ids,
                sample_values=sample_vals,
                severity=severity,
                description=f"Column '{col}' has {outlier_count} statistical/domain outliers outside [{lower_bound:.2f}, {upper_bound:.2f}].",
                detector="outlier_detector",
                evidence={
                    "method": "univariate_iqr_zscore",
                    "q25": q25,
                    "q75": q75,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "mean": mean,
                    "std": std,
                    "domain_bounds": {"min": domain_min, "max": domain_max},
                    "outlier_count": outlier_count
                }
            ))

    # =========================================================================
    # 3. MULTIVARIATE OUTLIER DETECTION (Isolation Forest)
    # =========================================================================
    if len(numeric_cols) >= 2 and total_rows >= 50:
        # Impute temporary median for NaN values in numeric matrix
        feature_matrix = clean_numeric_df[numeric_cols].copy()
        for c in numeric_cols:
            med_val = feature_matrix[c].median()
            feature_matrix[c] = feature_matrix[c].fillna(med_val if pd.notna(med_val) else 0.0)

        iso = IsolationForest(
            contamination=cfg.outlier_contamination,
            random_state=cfg.seed,
            n_estimators=100
        )
        preds = iso.fit_predict(feature_matrix)
        multi_mask = preds == -1
        multi_count = int(multi_mask.sum())

        if multi_count > 0:
            multi_row_ids = df_with_id.loc[multi_mask, "_row_id"].tolist()
            sample_rows = df_with_id.loc[multi_mask, numeric_cols].head(5).to_dict(orient="records")

            issues.append(Issue(
                issue_id="OUTLIER_MULTI_ISOFOREST",
                issue_type="outlier",
                columns=numeric_cols,
                row_ids=multi_row_ids,
                sample_values=sample_rows,
                severity="low",
                description=f"IsolationForest detected {multi_count} multivariate anomalous rows across {numeric_cols}.",
                detector="outlier_detector",
                evidence={
                    "method": "multivariate_isolation_forest",
                    "features": numeric_cols,
                    "contamination": cfg.outlier_contamination,
                    "anomaly_count": multi_count
                }
            ))

    return issues
