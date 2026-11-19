"""DataFix Confidence Scoring Engine: Computes confidence scores in [0, 1] for proposed repairs."""

from __future__ import annotations
from typing import Any, Dict, Optional
from datafix.config import Config
from datafix.models import Issue


def calculate_repair_confidence(
    issue: Issue,
    strategy: str,
    old_val: Any,
    new_val: Any,
    context: Optional[Dict[str, Any]] = None
) -> float:
    """Calculate a calibrated confidence score between 0.0 and 1.0 for a repair action."""
    ctx = context or {}
    strat = strategy.lower()

    # 1. Disguised Null Standardization (e.g. converting "N/A" -> np.nan or canonical empty)
    if strat == "standardize_disguised_null":
        return 0.99

    # 2. Whitespace and Casing Normalization
    if strat == "trim_whitespace":
        return 0.99
    if strat == "normalize_casing":
        return 0.95

    # 3. Currency / Numeric-as-text cleanup (stripping $, commas, USD to get pure float)
    if strat == "parse_numeric_string":
        return 0.98

    # 4. Canonical Category / Synonym Mapping (e.g. USA -> United States)
    if strat == "canonical_synonym_mapping":
        return 0.96

    if strat == "fuzzy_category_mapping":
        similarity = float(ctx.get("similarity", 0.85))
        # Scaled between 0.70 and 0.92
        if similarity >= 0.95:
            return 0.90
        elif similarity >= 0.90:
            return 0.85
        else:
            return 0.75  # Suggested only (< auto_apply_threshold 0.85)

    # 5. Date Format Standardization
    if strat == "standardize_date_format":
        is_valid_date = bool(ctx.get("is_valid_date", True))
        return 0.95 if is_valid_date else 0.60

    # 6. Duplicate row handling
    if strat == "deduplicate_exact_row":
        return 0.95
    if strat == "deduplicate_fuzzy_row":
        # Fuzzy duplicate merging carries ambiguity -> requires user confirmation
        sim = float(ctx.get("similarity", 0.85))
        return round(min(0.79, max(0.60, sim * 0.8)), 2)  # Suggested only

    # 7. Imputation
    if strat == "group_impute_median_mode":
        group_size = int(ctx.get("group_size", 10))
        return 0.88 if group_size >= 10 else 0.78
    if strat == "global_impute_median_mode":
        # Global imputation without grouping is speculative -> suggested
        return 0.75

    # 8. Relational repairs
    if strat == "recalculate_arithmetic_total":
        # When quantity and unit_price are valid positive numbers
        return 0.95

    if strat == "swap_inverted_dates":
        # Ambiguous which date was typed incorrectly -> suggested
        return 0.70

    if strat == "orphan_foreign_key_flag":
        # Cannot safely fabricate a foreign key -> suggested
        return 0.50

    if strat == "clamp_domain_outlier":
        return 0.85

    # Default fallback
    return 0.70
