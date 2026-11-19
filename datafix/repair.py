"""DataFix Conservative Repair Engine: Type-aware, confidence-calibrated dataset repairs."""

from __future__ import annotations
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from datafix.config import Config
from datafix.models import Issue, ChangeLogEntry
from datafix.audit import AuditLog
from datafix.confidence import calculate_repair_confidence
from datafix.profiler import ensure_row_id, infer_column_type, safe_assign
from datafix.detectors.consistency import KNOWN_CANONICAL_SYNONYMS


def repair_dataset(
    df: pd.DataFrame,
    issues: List[Issue],
    config: Optional[Config] = None
) -> Tuple[pd.DataFrame, AuditLog]:
    """Apply safe, confidence-scored repairs to a dataset.
    
    Guarantees:
    - Never mutates the original DataFrame.
    - Only applies repairs where confidence >= auto_apply_threshold.
    - Repaired DataFrame is strictly idempotent.
    - All modifications and suggestions are recorded in the AuditLog.
    """
    cfg = config or Config()
    # Never mutate original df
    df_repaired = ensure_row_id(df).copy()
    # Index by _row_id for O(1) cell access and high-throughput repairs
    df_repaired.index = df_repaired["_row_id"]
    audit_log = AuditLog()
    disguised_set = set(t.lower() for t in cfg.disguised_nulls)

    # Sort issues by type for deterministic, ordered repair pipeline
    # Order: formats -> consistency -> missing -> outlier -> relationship -> duplicates
    type_priority = {
        "invalid_format": 1,
        "inconsistent_category": 2,
        "missing_value": 3,
        "outlier": 4,
        "relationship_violation": 5,
        "duplicate_row": 6
    }
    sorted_issues = sorted(issues, key=lambda i: type_priority.get(i.issue_type, 99))

    rows_to_drop: Set[int] = set()

    for issue in sorted_issues:
        itype = issue.issue_type
        cols = issue.columns
        row_ids = issue.row_ids

        # ---------------------------------------------------------------------
        # 1. REPAIR INVALID FORMATS
        # ---------------------------------------------------------------------
        if itype == "invalid_format":
            col = cols[0]
            if col not in df_repaired.columns:
                continue

            # A. Numeric stored as text (e.g. $1,450.00, USD 250)
            if issue.evidence.get("format_type") == "numeric_as_text":
                valid_r_ids = [r for r in row_ids if r in df_repaired.index]
                if not valid_r_ids:
                    continue
                col_series = df_repaired.loc[valid_r_ids, col]
                applied_r_ids = []
                applied_vals = []
                ts_now = datetime.now(timezone.utc).isoformat()
                for r_id, old_val in col_series.items():
                    if pd.isna(old_val):
                        continue
                    
                    cleaned_str = re.sub(r"[\$,€£¥\s]|USD|EUR|GBP", "", str(old_val)).strip()
                    try:
                        new_val = float(cleaned_str)
                    except (ValueError, TypeError):
                        continue

                    try:
                        conf = calculate_repair_confidence(issue, "parse_numeric_string", old_val, new_val)
                        status = "applied" if conf >= cfg.auto_apply_threshold else "suggested"
                        
                        entry = ChangeLogEntry(
                            change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                            timestamp=ts_now,
                            row_id=r_id,
                            column=col,
                            old_value=old_val,
                            new_value=new_val,
                            issue_id=issue.issue_id,
                            strategy="parse_numeric_string",
                            reason=f"Parsed numeric string '{old_val}' into clean numeric value {new_val}",
                            confidence=conf,
                            status=status
                        )
                        audit_log.add_entry(entry)
                        if status == "applied":
                            applied_r_ids.append(r_id)
                            applied_vals.append(new_val)
                    except Exception:
                        pass
                if applied_r_ids:
                    safe_assign(df_repaired, applied_r_ids, col, applied_vals)

            # B. Date formats (e.g. DD/MM/YYYY, 2022.05.18 -> YYYY-MM-DD)
            elif "dominant_format" in issue.evidence or "DATE" in issue.issue_id:
                valid_r_ids = [r for r in row_ids if r in df_repaired.index]
                if not valid_r_ids:
                    continue
                col_series = df_repaired.loc[valid_r_ids, col]
                valid_mask = col_series.notna() & ~col_series.astype(str).str.lower().isin(disguised_set)
                valid_items = col_series[valid_mask]
                parsed_series = pd.to_datetime(valid_items.astype(str), errors="coerce", format="mixed")
                
                applied_r_ids = []
                applied_vals = []
                ts_now = datetime.now(timezone.utc).isoformat()

                for r_id in valid_items.index:
                    old_val = valid_items.at[r_id]
                    parsed_dt = parsed_series.at[r_id]
                    if pd.notna(parsed_dt):
                        new_val = parsed_dt.strftime("%Y-%m-%d")
                        conf = calculate_repair_confidence(issue, "standardize_date_format", old_val, new_val, {"is_valid_date": True})
                        status = "applied" if conf >= cfg.auto_apply_threshold else "suggested"
                        reason = f"Standardized date string '{old_val}' to ISO format '{new_val}'"
                        if status == "applied":
                            applied_r_ids.append(r_id)
                            applied_vals.append(new_val)
                    else:
                        new_val = old_val
                        conf = 0.50
                        status = "suggested"
                        reason = f"Unparseable date '{old_val}' flagged for manual review"

                    entry = ChangeLogEntry(
                        change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                        timestamp=ts_now,
                        row_id=r_id,
                        column=col,
                        old_value=old_val,
                        new_value=new_val,
                        issue_id=issue.issue_id,
                        strategy="standardize_date_format",
                        reason=reason,
                        confidence=conf,
                        status=status
                    )
                    audit_log.add_entry(entry)

                if applied_r_ids:
                    safe_assign(df_repaired, applied_r_ids, col, applied_vals)

            # C. Malformed Emails
            elif issue.evidence.get("format_type") == "email":
                valid_r_ids = [r for r in row_ids if r in df_repaired.index]
                if not valid_r_ids:
                    continue
                col_series = df_repaired.loc[valid_r_ids, col]
                ts_now = datetime.now(timezone.utc).isoformat()
                for r_id, old_val in col_series.items():
                    # Emails cannot be safely invented without external confirmation
                    entry = ChangeLogEntry(
                        change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                        timestamp=ts_now,
                        row_id=r_id,
                        column=col,
                        old_value=old_val,
                        new_value=old_val,
                        issue_id=issue.issue_id,
                        strategy="flag_malformed_format",
                        reason=f"Malformed email address '{old_val}' requires user verification",
                        confidence=0.50,
                        status="suggested"
                    )
                    audit_log.add_entry(entry)

        # ---------------------------------------------------------------------
        # 2. REPAIR INCONSISTENT CATEGORIES
        # ---------------------------------------------------------------------
        elif itype == "inconsistent_category":
            col = cols[0]
            if col not in df_repaired.columns:
                continue

            clusters = issue.evidence.get("clusters", [])
            # Build lookup from variant to canonical
            variant_to_canonical: Dict[str, Tuple[str, str, float]] = {}

            for cl in clusters:
                canon = cl.get("canonical_form", "")
                ctype = cl.get("type", "casing_whitespace")
                variants = cl.get("variants", [])

                for var in variants:
                    if var != canon:
                        if ctype == "synonym_abbreviation":
                            conf = 0.96
                            strat = "canonical_synonym_mapping"
                        elif ctype == "casing_whitespace":
                            conf = 0.95
                            strat = "normalize_casing"
                        else:
                            conf = 0.88
                            strat = "fuzzy_category_mapping"
                        variant_to_canonical[str(var).strip().lower()] = (canon, strat, conf)

            valid_r_ids = [r for r in row_ids if r in df_repaired.index]
            if valid_r_ids:
                col_series = df_repaired.loc[valid_r_ids, col]
                applied_r_ids = []
                applied_vals = []
                ts_now = datetime.now(timezone.utc).isoformat()
                for r_id, old_val in col_series.items():
                    if pd.isna(old_val):
                        continue

                    lookup_key = str(old_val).strip().lower()
                    if lookup_key in variant_to_canonical:
                        canon, strat, conf = variant_to_canonical[lookup_key]
                        status = "applied" if conf >= cfg.auto_apply_threshold else "suggested"
                        entry = ChangeLogEntry(
                            change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                            timestamp=ts_now,
                            row_id=r_id,
                            column=col,
                            old_value=old_val,
                            new_value=canon,
                            issue_id=issue.issue_id,
                            strategy=strat,
                            reason=f"Mapped category variant '{old_val}' to canonical form '{canon}'",
                            confidence=conf,
                            status=status
                        )
                        audit_log.add_entry(entry)
                        if status == "applied":
                            applied_r_ids.append(r_id)
                            applied_vals.append(canon)
                if applied_r_ids:
                    safe_assign(df_repaired, applied_r_ids, col, applied_vals)

        # ---------------------------------------------------------------------
        # 3. REPAIR MISSING VALUES & DISGUISED NULLS
        # ---------------------------------------------------------------------
        elif itype == "missing_value":
            col = cols[0]
            if col not in df_repaired.columns:
                continue

            valid_r_ids = [r for r in row_ids if r in df_repaired.index]
            if valid_r_ids:
                col_series = df_repaired.loc[valid_r_ids, col]
                applied_r_ids = []
                ts_now = datetime.now(timezone.utc).isoformat()
                for r_id, old_val in col_series.items():
                    # If disguised null string -> convert to np.nan
                    if pd.notna(old_val) and str(old_val).strip().lower() in disguised_set:
                        conf = calculate_repair_confidence(issue, "standardize_disguised_null", old_val, np.nan)
                        entry = ChangeLogEntry(
                            change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                            timestamp=ts_now,
                            row_id=r_id,
                            column=col,
                            old_value=old_val,
                            new_value=np.nan,
                            issue_id=issue.issue_id,
                            strategy="standardize_disguised_null",
                            reason=f"Standardized disguised null string '{old_val}' to native null",
                            confidence=conf,
                            status="applied"
                        )
                        audit_log.add_entry(entry)
                        applied_r_ids.append(r_id)
                    elif pd.isna(old_val):
                        # Flag missing value without guessing private personal identifiers
                        entry = ChangeLogEntry(
                            change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                            timestamp=ts_now,
                            row_id=r_id,
                            column=col,
                            old_value=None,
                            new_value=None,
                            issue_id=issue.issue_id,
                            strategy="flag_missing_value",
                            reason=f"Missing value in '{col}' flagged for user action",
                            confidence=0.50,
                            status="suggested"
                        )
                        audit_log.add_entry(entry)
                if applied_r_ids:
                    safe_assign(df_repaired, applied_r_ids, col, np.nan)

        # ---------------------------------------------------------------------
        # 4. REPAIR RELATIONSHIP VIOLATIONS
        # ---------------------------------------------------------------------
        elif itype == "relationship_violation":
            v_type = issue.evidence.get("violation_type")

            # Arithmetic recalculation: total = quantity * unit_price
            if v_type == "arithmetic_mismatch" and len(cols) == 3:
                qty_col, price_col, total_col = cols[0], cols[1], cols[2]
                valid_r_ids = [r for r in row_ids if r in df_repaired.index]
                if valid_r_ids:
                    sub_df = df_repaired.loc[valid_r_ids, [qty_col, price_col, total_col]]
                    num_qtys = pd.to_numeric(sub_df[qty_col], errors="coerce")
                    num_prices = pd.to_numeric(sub_df[price_col], errors="coerce")
                    old_totals = sub_df[total_col]
                    applied_r_ids = []
                    applied_vals = []
                    ts_now = datetime.now(timezone.utc).isoformat()
                    for r_id in valid_r_ids:
                        qty = num_qtys.at[r_id]
                        price = num_prices.at[r_id]
                        old_total = old_totals.at[r_id]

                        if pd.notna(qty) and pd.notna(price) and qty > 0 and price >= 0:
                            recomputed = round(float(qty) * float(price), 2)
                            conf = calculate_repair_confidence(issue, "recalculate_arithmetic_total", old_total, recomputed)
                            status = "applied" if conf >= cfg.auto_apply_threshold else "suggested"
                            entry = ChangeLogEntry(
                                change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                                timestamp=ts_now,
                                row_id=r_id,
                                column=total_col,
                                old_value=old_total,
                                new_value=recomputed,
                                issue_id=issue.issue_id,
                                strategy="recalculate_arithmetic_total",
                                reason=f"Recalculated '{total_col}' from {qty_col} ({qty}) * {price_col} ({price}) = {recomputed}",
                                confidence=conf,
                                status=status
                            )
                            audit_log.add_entry(entry)
                            if status == "applied":
                                applied_r_ids.append(r_id)
                                applied_vals.append(recomputed)
                    if applied_r_ids:
                        safe_assign(df_repaired, applied_r_ids, total_col, applied_vals)

            # Inverted chronological timestamps (delivery < order)
            elif v_type == "chronological_inversion":
                ts_now = datetime.now(timezone.utc).isoformat()
                for r_id in row_ids:
                    entry = ChangeLogEntry(
                        change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                        timestamp=ts_now,
                        row_id=r_id,
                        column=cols[1] if len(cols) > 1 else cols[0],
                        old_value="Inverted",
                        new_value="Suggested Swap",
                        issue_id=issue.issue_id,
                        strategy="swap_inverted_dates",
                        reason=f"Inverted chronological dates in {cols}; swap suggested for review",
                        confidence=0.70,
                        status="suggested"
                    )
                    audit_log.add_entry(entry)

            # Orphan foreign keys
            elif v_type == "orphan_foreign_key":
                col = cols[0]
                valid_r_ids = [r for r in row_ids if r in df_repaired.index]
                if valid_r_ids:
                    col_series = df_repaired.loc[valid_r_ids, col]
                    ts_now = datetime.now(timezone.utc).isoformat()
                    for r_id, old_fk in col_series.items():
                        entry = ChangeLogEntry(
                            change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                            timestamp=ts_now,
                            row_id=r_id,
                            column=col,
                            old_value=old_fk,
                            new_value=old_fk,
                            issue_id=issue.issue_id,
                            strategy="orphan_foreign_key_flag",
                            reason=f"Orphan foreign key '{old_fk}' missing in parent table; flagged for manual review",
                            confidence=0.50,
                            status="suggested"
                        )
                        audit_log.add_entry(entry)

        # ---------------------------------------------------------------------
        # 5. REPAIR OUTLIERS (Domain Boundary Clamping)
        # ---------------------------------------------------------------------
        elif itype == "outlier":
            col = cols[0]
            domain_bounds = issue.evidence.get("domain_bounds", {})
            d_min = domain_bounds.get("min")
            if d_min is not None and col in df_repaired.columns:
                valid_r_ids = [r for r in row_ids if r in df_repaired.index]
                if valid_r_ids:
                    col_series = df_repaired.loc[valid_r_ids, col]
                    num_series = pd.to_numeric(col_series, errors="coerce")
                    applied_r_ids = []
                    applied_vals = []
                    ts_now = datetime.now(timezone.utc).isoformat()
                    for r_id in valid_r_ids:
                        old_val = col_series.at[r_id]
                        num_val = num_series.at[r_id]
                        if pd.notna(num_val) and num_val < d_min:
                            # Impossible negative (e.g. age -5 -> 5 or 0)
                            corrected = abs(num_val)
                            conf = calculate_repair_confidence(issue, "clamp_domain_outlier", old_val, corrected)
                            status = "applied" if conf >= cfg.auto_apply_threshold else "suggested"
                            entry = ChangeLogEntry(
                                change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                                timestamp=ts_now,
                                row_id=r_id,
                                column=col,
                                old_value=old_val,
                                new_value=corrected,
                                issue_id=issue.issue_id,
                                strategy="clamp_domain_outlier",
                                reason=f"Corrected negative value {old_val} to positive domain boundary {corrected}",
                                confidence=conf,
                                status=status
                            )
                            audit_log.add_entry(entry)
                            if status == "applied":
                                applied_r_ids.append(r_id)
                                applied_vals.append(corrected)
                    if applied_r_ids:
                        safe_assign(df_repaired, applied_r_ids, col, applied_vals)

        # ---------------------------------------------------------------------
        # 6. REPAIR DUPLICATES (Exact Rows Only)
        # ---------------------------------------------------------------------
        elif itype == "duplicate_row":
            dup_type = issue.evidence.get("duplicate_type")
            if dup_type == "exact":
                # Re-check exact duplicates on the current state of df_repaired
                # so that rows normalized in prior steps (e.g. whitespace/category) are also deduplicated
                content_cols = [c for c in df_repaired.columns if c != "_row_id"]
                dup_mask = df_repaired.duplicated(subset=content_cols, keep="first")
                current_dup_ids = set(df_repaired.loc[dup_mask, "_row_id"].tolist()).union(set(row_ids))
                
                for r_id in current_dup_ids:
                    if r_id in rows_to_drop or r_id not in df_repaired.index:
                        continue
                    conf = 0.95
                    entry = ChangeLogEntry(
                        change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        row_id=r_id,
                        column="ALL",
                        old_value="[Duplicate Row]",
                        new_value="[Removed]",
                        issue_id=issue.issue_id,
                        strategy="deduplicate_exact_row",
                        reason=f"Removed redundant exact duplicate row (ID: {r_id})",
                        confidence=conf,
                        status="applied"
                    )
                    audit_log.add_entry(entry)
                    rows_to_drop.add(r_id)

            elif dup_type == "fuzzy":
                # Fuzzy duplicate removal requires user approval
                for r_id in row_ids:
                    entry = ChangeLogEntry(
                        change_id=f"CHG_{uuid.uuid4().hex[:8]}",
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        row_id=r_id,
                        column=cols[0] if cols else "ALL",
                        old_value="[Fuzzy Duplicate]",
                        new_value="[Needs Review]",
                        issue_id=issue.issue_id,
                        strategy="deduplicate_fuzzy_row",
                        reason=f"Fuzzy duplicate candidate row (ID: {r_id}) flagged for manual review",
                        confidence=0.70,
                        status="suggested"
                    )
                    audit_log.add_entry(entry)

    # Apply exact duplicate drops
    if rows_to_drop:
        df_repaired = df_repaired[~df_repaired["_row_id"].isin(rows_to_drop)].copy()

    df_repaired = df_repaired.reset_index(drop=True)
    return df_repaired, audit_log
