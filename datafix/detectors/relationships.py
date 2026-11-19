"""Detector for relational violations: orphan foreign keys, date ordering, arithmetic consistency."""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from datafix.config import Config
from datafix.models import Issue
from datafix.profiler import ensure_row_id


DATE_PAIR_CANDIDATES = [
    ("order_date", "delivery_date"),
    ("start_date", "end_date"),
    ("created_at", "updated_at"),
    ("signup_date", "delivery_date"),
    ("departure_date", "arrival_date"),
    ("checkin_date", "checkout_date"),
]


def detect_relationship_violations(
    df: pd.DataFrame,
    ref_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None
) -> List[Issue]:
    """Detect broken relationships: orphan foreign keys, inverted date intervals, arithmetic mismatches."""
    cfg = config or Config()
    df_with_id = ensure_row_id(df)
    issues: List[Issue] = []

    total_rows = len(df_with_id)
    if total_rows == 0:
        return issues

    cols = df_with_id.columns
    cols_lower_map = {c.lower(): c for c in cols}

    # =========================================================================
    # 1. ORPHAN FOREIGN KEYS (Multi-table)
    # =========================================================================
    if ref_dfs:
        for ref_name, ref_df in ref_dfs.items():
            if ref_df is None or len(ref_df) == 0:
                continue
            ref_cols_lower = {c.lower(): c for c in ref_df.columns}

            # Check for shared key columns (e.g. customer_id, user_id, product_id)
            for c_lower, c_orig in cols_lower_map.items():
                if c_lower.endswith(("_id", "id", "_key", "key")) and c_lower in ref_cols_lower:
                    ref_c_orig = ref_cols_lower[c_lower]
                    
                    # Non-null values in current df
                    curr_keys = df_with_id[c_orig].dropna()
                    if curr_keys.empty:
                        continue

                    valid_ref_keys = set(ref_df[ref_c_orig].dropna().unique())
                    orphan_mask = ~df_with_id[c_orig].isna() & (~df_with_id[c_orig].isin(valid_ref_keys))
                    orphan_count = int(orphan_mask.sum())

                    if orphan_count > 0:
                        orphan_row_ids = df_with_id.loc[orphan_mask, "_row_id"].tolist()
                        sample_vals = df_with_id.loc[orphan_mask, c_orig].head(5).tolist()

                        issues.append(Issue(
                            issue_id=f"REL_ORPHAN_{c_orig}_{ref_name}",
                            issue_type="relationship_violation",
                            columns=[c_orig],
                            row_ids=orphan_row_ids,
                            sample_values=sample_vals,
                            severity="high",
                            description=f"Found {orphan_count} orphan rows in '{c_orig}' referencing missing parent keys in '{ref_name}.{ref_c_orig}'.",
                            detector="relationship_detector",
                            evidence={
                                "violation_type": "orphan_foreign_key",
                                "foreign_key_column": c_orig,
                                "parent_table": ref_name,
                                "parent_column": ref_c_orig,
                                "orphan_count": orphan_count
                            }
                        ))

    # =========================================================================
    # 2. CROSS-COLUMN DATE ORDERING (end_date < start_date)
    # =========================================================================
    for start_col_cand, end_col_cand in DATE_PAIR_CANDIDATES:
        if start_col_cand in cols_lower_map and end_col_cand in cols_lower_map:
            c_start = cols_lower_map[start_col_cand]
            c_end = cols_lower_map[end_col_cand]

            dt_start = pd.to_datetime(df_with_id[c_start], errors="coerce", format="mixed")
            dt_end = pd.to_datetime(df_with_id[c_end], errors="coerce", format="mixed")

            # Condition: both dates are valid, but end_date < start_date
            inverted_mask = dt_start.notna() & dt_end.notna() & (dt_end < dt_start)
            inverted_count = int(inverted_mask.sum())

            if inverted_count > 0:
                bad_row_ids = df_with_id.loc[inverted_mask, "_row_id"].tolist()
                sample_pairs = [
                    {"start": str(s), "end": str(e)}
                    for s, e in zip(df_with_id.loc[inverted_mask, c_start].head(5), df_with_id.loc[inverted_mask, c_end].head(5))
                ]

                issues.append(Issue(
                    issue_id=f"REL_DATE_ORDER_{c_start}_{c_end}",
                    issue_type="relationship_violation",
                    columns=[c_start, c_end],
                    row_ids=bad_row_ids,
                    sample_values=sample_pairs,
                    severity="high",
                    description=f"Cross-column violation: '{c_end}' occurs before '{c_start}' in {inverted_count} rows.",
                    detector="relationship_detector",
                    evidence={
                        "violation_type": "chronological_inversion",
                        "start_column": c_start,
                        "end_column": c_end,
                        "inversion_count": inverted_count
                    }
                ))

    # =========================================================================
    # 3. CROSS-COLUMN ARITHMETIC RULES (total != quantity * unit_price)
    # =========================================================================
    qty_col = next((cols_lower_map[k] for k in ["quantity", "qty", "units"] if k in cols_lower_map), None)
    price_col = next((cols_lower_map[k] for k in ["unit_price", "price", "rate"] if k in cols_lower_map), None)
    total_col = next((cols_lower_map[k] for k in ["total_amount", "total", "amount", "line_total"] if k in cols_lower_map and k != price_col), None)

    if qty_col and price_col and total_col:
        num_qty = pd.to_numeric(df_with_id[qty_col], errors="coerce")
        num_price = pd.to_numeric(
            df_with_id[price_col].astype(str).str.replace(r"[\$,€£¥\s]", "", regex=True),
            errors="coerce"
        )
        num_total = pd.to_numeric(
            df_with_id[total_col].astype(str).str.replace(r"[\$,€£¥\s]", "", regex=True),
            errors="coerce"
        )

        valid_calc_mask = num_qty.notna() & num_price.notna() & num_total.notna()
        if valid_calc_mask.any():
            expected_total = num_qty * num_price
            # Check difference exceeds tolerance (0.05 for rounding)
            diff = (num_total - expected_total).abs()
            mismatch_mask = valid_calc_mask & (diff > 0.05)
            mismatch_count = int(mismatch_mask.sum())

            if mismatch_count > 0:
                bad_row_ids = df_with_id.loc[mismatch_mask, "_row_id"].tolist()
                sample_calc = [
                    {
                        "qty": q,
                        "price": p,
                        "actual_total": t,
                        "expected_total": round(q * p, 2)
                    }
                    for q, p, t in zip(
                        num_qty.loc[mismatch_mask].head(5),
                        num_price.loc[mismatch_mask].head(5),
                        num_total.loc[mismatch_mask].head(5)
                    )
                ]

                issues.append(Issue(
                    issue_id=f"REL_CALC_{total_col}",
                    issue_type="relationship_violation",
                    columns=[qty_col, price_col, total_col],
                    row_ids=bad_row_ids,
                    sample_values=sample_calc,
                    severity="high",
                    description=f"Arithmetic mismatch: '{total_col}' does not equal '{qty_col}' * '{price_col}' in {mismatch_count} rows.",
                    detector="relationship_detector",
                    evidence={
                        "violation_type": "arithmetic_mismatch",
                        "formula": f"{total_col} == {qty_col} * {price_col}",
                        "mismatch_count": mismatch_count
                    }
                ))

    return issues
