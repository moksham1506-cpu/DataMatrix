"""Detector for exact and fuzzy duplicate rows."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from datafix.config import Config
from datafix.models import Issue
from datafix.profiler import ensure_row_id


def detect_duplicates(df: pd.DataFrame, config: Optional[Config] = None) -> List[Issue]:
    """Detect exact and fuzzy duplicate rows in a DataFrame."""
    cfg = config or Config()
    df_with_id = ensure_row_id(df)
    issues: List[Issue] = []
    
    total_rows = len(df_with_id)
    if total_rows < 2:
        return issues

    content_cols = [c for c in df_with_id.columns if c != "_row_id"]
    if not content_cols:
        return issues

    # =========================================================================
    # 1. EXACT DUPLICATES
    # =========================================================================
    # Find all rows that share identical values across content columns
    dup_mask = df_with_id.duplicated(subset=content_cols, keep=False)
    if dup_mask.any():
        dup_df = df_with_id[dup_mask]
        # Group identical rows together
        exact_groups: List[List[int]] = []
        grouped = dup_df.groupby(content_cols)
        
        all_dup_row_ids: List[int] = []
        sample_rows: List[Any] = []

        for _, group in grouped:
            row_ids = group["_row_id"].tolist()
            if len(row_ids) > 1:
                exact_groups.append(row_ids)
                # First row is considered canonical, subsequent are duplicate instances
                all_dup_row_ids.extend(row_ids[1:])
                sample_rows.append(group.iloc[0][content_cols].to_dict())

        if all_dup_row_ids:
            issues.append(Issue(
                issue_id="DUP_EXACT",
                issue_type="duplicate_row",
                columns=content_cols,
                row_ids=all_dup_row_ids,
                sample_values=sample_rows[:5],
                severity="high",
                description=f"Found {len(all_dup_row_ids)} exact duplicate rows across {len(exact_groups)} distinct duplicate clusters.",
                detector="duplicate_detector",
                evidence={
                    "duplicate_type": "exact",
                    "duplicate_count": len(all_dup_row_ids),
                    "cluster_count": len(exact_groups),
                    "duplicate_groups": exact_groups[:20]
                }
            ))

    # =========================================================================
    # 2. FUZZY DUPLICATES
    # =========================================================================
    # Identify candidate text columns for fuzzy matching (name, address, company, etc.)
    entity_keywords = ["name", "title", "company", "customer", "client", "vendor", "address", "street", "person"]
    text_cols = [
        c for c in content_cols
        if df_with_id[c].dtype == object or isinstance(df_with_id[c].dtype, pd.StringDtype)
    ]
    name_cols = [c for c in text_cols if any(k in c.lower() for k in entity_keywords)]
    target_cols = name_cols

    if target_cols:
        # Create normalized string representation for fuzzy comparison
        # Use blocking by first 2 characters / length to avoid O(N^2) explosion
        threshold = cfg.similarity_threshold * 100.0
        fuzzy_pairs: List[Tuple[int, int, float]] = []
        fuzzy_row_ids: Set[int] = set()

        # Build concatenated comparison string
        text_series = df_with_id[target_cols[0]].fillna("").astype(str).str.strip().str.lower()
        
        # Exclude exact duplicates already found
        excluded_ids = set(all_dup_row_ids) if dup_mask.any() else set()
        valid_indices = [
            (row_id, text)
            for row_id, text in zip(df_with_id["_row_id"], text_series)
            if row_id not in excluded_ids and len(text) >= 4
        ]

        # Block by first 2 characters of text for high-performance indexing
        blocks = defaultdict(list)
        for r_id, txt in valid_indices:
            block_key = txt[:2]
            blocks[block_key].append((r_id, txt))

        for block_key, items in blocks.items():
            n_items = len(items)
            if n_items < 2 or n_items > 300:
                continue
            for i in range(n_items):
                r_id1, txt1 = items[i]
                for j in range(i + 1, n_items):
                    r_id2, txt2 = items[j]
                    if txt1 == txt2:
                        continue
                    sim = fuzz.ratio(txt1, txt2)
                    if sim >= max(threshold, 93.0):
                        fuzzy_pairs.append((r_id1, r_id2, round(sim / 100.0, 3)))
                        fuzzy_row_ids.add(r_id2)

        if fuzzy_pairs:
            sample_samples = []
            for r1, r2, s in fuzzy_pairs[:5]:
                val1 = df_with_id.loc[df_with_id["_row_id"] == r1, target_cols[0]].values[0]
                val2 = df_with_id.loc[df_with_id["_row_id"] == r2, target_cols[0]].values[0]
                sample_samples.append({"original": val1, "fuzzy_variant": val2, "similarity": s})

            issues.append(Issue(
                issue_id="DUP_FUZZY",
                issue_type="duplicate_row",
                columns=target_cols,
                row_ids=sorted(list(fuzzy_row_ids)),
                sample_values=sample_samples,
                severity="med",
                description=f"Found {len(fuzzy_pairs)} fuzzy duplicate pairs on {target_cols} with similarity >= {threshold:.0f}%.",
                detector="duplicate_detector",
                evidence={
                    "duplicate_type": "fuzzy",
                    "similarity_threshold": cfg.similarity_threshold,
                    "target_columns": target_cols,
                    "fuzzy_pairs": [{"row_id_1": r1, "row_id_2": r2, "similarity": s} for r1, r2, s in fuzzy_pairs[:25]]
                }
            ))

    return issues
