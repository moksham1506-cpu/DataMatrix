"""Detector for inconsistent categories (casing, whitespace, abbreviation, fuzzy variants)."""

from __future__ import annotations
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from datafix.config import Config
from datafix.models import Issue
from datafix.profiler import ensure_row_id, infer_column_type


KNOWN_CANONICAL_SYNONYMS = {
    "united states": ["usa", "u.s.a.", "u.s.", "us", "united states of america"],
    "united kingdom": ["uk", "u.k.", "great britain", "britain"],
    "canada": ["can", "ca"],
    "germany": ["de", "deu", "deutschland"],
    "france": ["fr", "fra"],
    "australia": ["au", "aus"],
}


def detect_inconsistencies(df: pd.DataFrame, config: Optional[Config] = None) -> List[Issue]:
    """Detect casing, whitespace, and fuzzy/abbreviation category inconsistencies."""
    cfg = config or Config()
    df_with_id = ensure_row_id(df)
    issues: List[Issue] = []

    total_rows = len(df_with_id)
    if total_rows == 0:
        return issues

    disguised_set = set(token.lower() for token in cfg.disguised_nulls)
    content_cols = [c for c in df_with_id.columns if c != "_row_id"]

    for col in content_cols:
        series = df_with_id[col]
        # Only inspect object/string columns
        if series.dtype != object and not isinstance(series.dtype, pd.StringDtype):
            continue

        col_lower = col.lower()
        # Skip date, numeric, identifier columns from category consistency checks
        if any(kw in col_lower for kw in ["date", "time", "_id", "id", "_key", "key", "timestamp", "price", "amount", "phone", "email"]):
            continue

        inferred = infer_column_type(series, disguised_set)
        if inferred not in {"category", "text"}:
            continue

        valid_mask = series.notna()
        if not valid_mask.any():
            continue

        raw_strings = series[valid_mask].astype(str)
        # Filter disguised nulls
        non_null_mask = ~raw_strings.str.strip().str.lower().isin(disguised_set) & (raw_strings.str.strip() != "")
        clean_raw = raw_strings[non_null_mask]
        if clean_raw.empty:
            continue

        # Check unique ratio (only for categorical columns with reasonable cardinality)
        unique_count = clean_raw.nunique()
        if unique_count > 200 or (unique_count / len(clean_raw) > 0.40 and unique_count > 30):
            continue

        # 1. Check casing and whitespace variants
        # Group by stripped lowercase form
        val_counts = clean_raw.value_counts()
        normalized_map = defaultdict(list)
        for raw_val, count in val_counts.items():
            norm_key = " ".join(raw_val.strip().lower().split())
            normalized_map[norm_key].append((raw_val, count))

        case_whitespace_issues: Dict[str, List[Tuple[str, int]]] = {}
        for norm_key, variants in normalized_map.items():
            if len(variants) > 1:
                # Multiple surface representations of the same normalized token
                case_whitespace_issues[norm_key] = variants

        # 2. Check known synonyms / abbreviations (e.g. USA vs United States)
        synonym_clusters: List[Dict[str, Any]] = []
        unique_tokens = list(normalized_map.keys())
        token_set = set(unique_tokens)

        for canonical, aliases in KNOWN_CANONICAL_SYNONYMS.items():
            present_aliases = [a for a in aliases if a in token_set]
            has_canonical = canonical in token_set
            if present_aliases and (has_canonical or len(present_aliases) > 1):
                synonym_clusters.append({
                    "canonical": canonical.title(),
                    "variants": present_aliases + ([canonical] if has_canonical else [])
                })

        # 3. Check fuzzy category variants (rapidfuzz token_sort_ratio >= category_similarity_threshold)
        fuzzy_clusters: List[Dict[str, Any]] = []
        n_tokens = len(unique_tokens)
        sim_threshold = cfg.category_similarity_threshold * 100.0

        visited = set()
        for i in range(n_tokens):
            t1 = unique_tokens[i]
            if t1 in visited or len(t1) < 4:
                continue
            cluster = [t1]
            for j in range(i + 1, n_tokens):
                t2 = unique_tokens[j]
                if t2 in visited or len(t2) < 4:
                    continue
                # Calculate similarity
                sim = fuzz.token_sort_ratio(t1, t2)
                if sim >= sim_threshold:
                    cluster.append(t2)
                    visited.add(t2)
            if len(cluster) > 1:
                visited.add(t1)
                # Pick the most frequent token as candidate
                cand = max(cluster, key=lambda t: sum(cnt for _, cnt in normalized_map[t]))
                fuzzy_clusters.append({
                    "canonical": cand,
                    "variants": cluster
                })

        # Collate issue if any variants found
        all_violating_row_ids: Set[int] = set()
        all_variant_details: List[Dict[str, Any]] = []

        # Find row ids for casing/whitespace variants
        for norm_key, variants in case_whitespace_issues.items():
            # Canonical is the most common variant
            dominant_var = max(variants, key=lambda x: x[1])[0]
            for var, cnt in variants:
                if var != dominant_var:
                    bad_idx = clean_raw[clean_raw == var].index
                    bad_row_ids = df_with_id.loc[bad_idx, "_row_id"].tolist()
                    all_violating_row_ids.update(bad_row_ids)
            all_variant_details.append({
                "type": "casing_whitespace",
                "canonical_form": dominant_var,
                "variants": [v for v, _ in variants]
            })

        # Find row ids for synonym / abbreviation variants
        for syn in synonym_clusters:
            canon = syn["canonical"]
            variants = syn["variants"]
            for var in variants:
                # If variant is not already matching canonical
                if var.lower() != canon.lower():
                    # Match any raw string whose normalized form is var
                    matching_raws = [raw_v for raw_v, _ in normalized_map.get(var, [])]
                    for mr in matching_raws:
                        bad_idx = clean_raw[clean_raw == mr].index
                        bad_row_ids = df_with_id.loc[bad_idx, "_row_id"].tolist()
                        all_violating_row_ids.update(bad_row_ids)
            all_variant_details.append({
                "type": "synonym_abbreviation",
                "canonical_form": canon,
                "variants": variants
            })

        # Find row ids for fuzzy clusters
        for fz in fuzzy_clusters:
            canon = fz["canonical"]
            for var in fz["variants"]:
                if var != canon:
                    matching_raws = [raw_v for raw_v, _ in normalized_map.get(var, [])]
                    for mr in matching_raws:
                        bad_idx = clean_raw[clean_raw == mr].index
                        bad_row_ids = df_with_id.loc[bad_idx, "_row_id"].tolist()
                        all_violating_row_ids.update(bad_row_ids)
            all_variant_details.append({
                "type": "fuzzy_variant",
                "canonical_form": canon,
                "variants": fz["variants"]
            })

        if all_violating_row_ids:
            sorted_row_ids = sorted(list(all_violating_row_ids))
            sample_vals = series.loc[df_with_id["_row_id"].isin(sorted_row_ids[:5])].tolist()
            issues.append(Issue(
                issue_id=f"INCONSISTENT_{col}",
                issue_type="inconsistent_category",
                columns=[col],
                row_ids=sorted_row_ids,
                sample_values=sample_vals,
                severity="med",
                description=f"Column '{col}' has {len(sorted_row_ids)} inconsistent category variants across {len(all_variant_details)} pattern clusters.",
                detector="consistency_detector",
                evidence={
                    "total_inconsistent_rows": len(sorted_row_ids),
                    "clusters": all_variant_details
                }
            ))

    return issues
