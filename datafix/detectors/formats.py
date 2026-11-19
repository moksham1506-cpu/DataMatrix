"""Detector for invalid or inconsistent data formats (dates, emails, phones, numerics)."""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from datafix.config import Config
from datafix.models import Issue
from datafix.profiler import ensure_row_id, infer_column_type, EMAIL_REGEX, PHONE_REGEX


# Regex for detecting currency/formatted numeric strings
CURRENCY_REGEX = re.compile(r"[$€£¥]|USD|EUR|GBP", re.IGNORECASE)
NUMERIC_CLEAN_REGEX = re.compile(r"^[\s$€£¥A-Za-z,]*[-+]?[0-9]+(\.[0-9]+)?[\s%]*$")

# Common date format regex patterns
ISO_DATE_REGEX = re.compile(r"^\d{4}-\d{2}-\d{2}$")
SLASH_DATE_REGEX = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
DOT_DATE_REGEX = re.compile(r"^\d{4}\.\d{2}\.\d{2}$")
TEXT_DATE_REGEX = re.compile(r"^[A-Za-z]{3}\s+\d{1,2},\s+\d{4}$")


def detect_format_issues(df: pd.DataFrame, config: Optional[Config] = None) -> List[Issue]:
    """Detect invalid and inconsistent formats across dates, emails, phones, and numerics."""
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
        # Skip purely null columns
        valid_mask = series.notna()
        if not valid_mask.any():
            continue

        clean_series = series[valid_mask].astype(str).str.strip()
        # Filter out disguised nulls so they aren't double-flagged as format errors
        non_null_mask = ~clean_series.str.lower().isin(disguised_set) & (clean_series != "")
        sub_series = clean_series[non_null_mask]
        if sub_series.empty:
            continue

        inferred = infer_column_type(series, disguised_set)

        # ---------------------------------------------------------------------
        # 1. DATE FORMAT DETECTION (Mixed formats & Invalid calendar dates)
        # ---------------------------------------------------------------------
        is_id_col = col.lower().endswith(("_id", "id", "_key", "key"))
        is_date_col = (inferred == "datetime" or any(kw in col.lower() for kw in ["date", "time", "signup", "created", "timestamp"])) and not is_id_col
        if is_date_col:
            date_strings = sub_series.astype(str)
            # Check format patterns
            iso_matches = date_strings.map(lambda x: bool(ISO_DATE_REGEX.match(x)))
            slash_matches = date_strings.map(lambda x: bool(SLASH_DATE_REGEX.match(x)))
            dot_matches = date_strings.map(lambda x: bool(DOT_DATE_REGEX.match(x)))
            text_matches = date_strings.map(lambda x: bool(TEXT_DATE_REGEX.match(x)))

            # Parse validity with pd.to_datetime
            parsed_dates = pd.to_datetime(date_strings, errors="coerce", format="mixed")
            unparseable_mask = parsed_dates.isna()
            
            # If the column has multiple different date format conventions
            format_counts = {
                "YYYY-MM-DD": int(iso_matches.sum()),
                "DD/MM/YYYY": int(slash_matches.sum()),
                "YYYY.MM.DD": int(dot_matches.sum()),
                "Mon DD, YYYY": int(text_matches.sum()),
                "unparseable": int(unparseable_mask.sum())
            }
            # Identify dominant format
            known_formats = {k: v for k, v in format_counts.items() if k != "unparseable" and v > 0}
            
            # Only trigger if the column actually has recognizable dates (at least 20% parseable)
            parseable_count = sum(known_formats.values())
            if parseable_count / len(sub_series) >= 0.20:
                if len(known_formats) > 1 or unparseable_mask.any():
                    dominant_fmt = max(known_formats.items(), key=lambda x: x[1])[0] if known_formats else "YYYY-MM-DD"
                    
                    # Flag rows that don't match dominant format or are unparseable
                    if dominant_fmt == "YYYY-MM-DD":
                        non_conforming = ~iso_matches | unparseable_mask
                    elif dominant_fmt == "DD/MM/YYYY":
                        non_conforming = ~slash_matches | unparseable_mask
                    else:
                        non_conforming = unparseable_mask

                    bad_indices = sub_series[non_conforming].index
                    if len(bad_indices) > 0:
                        bad_row_ids = df_with_id.loc[bad_indices, "_row_id"].tolist()
                        sample_vals = series.loc[bad_indices].head(5).tolist()
                        issues.append(Issue(
                            issue_id=f"FMT_DATE_{col}",
                            issue_type="invalid_format",
                            columns=[col],
                            row_ids=bad_row_ids,
                            sample_values=sample_vals,
                            severity="med" if len(bad_row_ids) / len(sub_series) < 0.20 else "high",
                            description=f"Column '{col}' has {len(bad_row_ids)} date format irregularities. Dominant format: {dominant_fmt}.",
                            detector="format_detector",
                            evidence={
                                "dominant_format": dominant_fmt,
                                "format_breakdown": format_counts,
                                "irregular_count": len(bad_row_ids),
                                "unparseable_count": int(unparseable_mask.sum())
                            }
                        ))

        # ---------------------------------------------------------------------
        # 2. EMAIL FORMAT DETECTION
        # ---------------------------------------------------------------------
        elif inferred == "email" or "email" in col.lower():
            email_matches = sub_series.map(lambda x: bool(EMAIL_REGEX.match(x)))
            bad_email_mask = ~email_matches
            if bad_email_mask.any():
                bad_indices = sub_series[bad_email_mask].index
                bad_row_ids = df_with_id.loc[bad_indices, "_row_id"].tolist()
                sample_vals = series.loc[bad_indices].head(5).tolist()
                issues.append(Issue(
                    issue_id=f"FMT_EMAIL_{col}",
                    issue_type="invalid_format",
                    columns=[col],
                    row_ids=bad_row_ids,
                    sample_values=sample_vals,
                    severity="high",
                    description=f"Column '{col}' contains {len(bad_row_ids)} malformed email addresses.",
                    detector="format_detector",
                    evidence={
                        "format_type": "email",
                        "malformed_count": len(bad_row_ids),
                        "total_inspected": len(sub_series)
                    }
                ))

        # ---------------------------------------------------------------------
        # 3. PHONE NUMBER FORMAT DETECTION
        # ---------------------------------------------------------------------
        elif inferred == "phone" or any(kw in col.lower() for kw in ["phone", "mobile", "cell", "tel"]):
            phone_matches = sub_series.map(lambda x: bool(PHONE_REGEX.match(x)))
            # Additionally check if contains letters or invalid characters
            has_letters = sub_series.str.contains(r"[a-zA-Z]", regex=True)
            bad_phone_mask = (~phone_matches) | has_letters
            if bad_phone_mask.any():
                bad_indices = sub_series[bad_phone_mask].index
                bad_row_ids = df_with_id.loc[bad_indices, "_row_id"].tolist()
                sample_vals = series.loc[bad_indices].head(5).tolist()
                issues.append(Issue(
                    issue_id=f"FMT_PHONE_{col}",
                    issue_type="invalid_format",
                    columns=[col],
                    row_ids=bad_row_ids,
                    sample_values=sample_vals,
                    severity="med",
                    description=f"Column '{col}' contains {len(bad_row_ids)} invalid phone numbers with malformed characters or lengths.",
                    detector="format_detector",
                    evidence={
                        "format_type": "phone",
                        "malformed_count": len(bad_row_ids),
                        "has_alphabetic_characters": int(has_letters.sum())
                    }
                ))

        # ---------------------------------------------------------------------
        # 4. NUMERICS STORED AS TEXT / CURRENCY
        # ---------------------------------------------------------------------
        elif (series.dtype == object or isinstance(series.dtype, pd.StringDtype)) and (
            inferred == "numeric" or any(kw in col.lower() for kw in ["amount", "price", "cost", "total", "fee", "salary", "balance"])
        ):
            # Check for currency symbols, commas, or "USD"
            has_currency = sub_series.str.contains(CURRENCY_REGEX, regex=True)
            has_commas = sub_series.str.contains(r"\d+,\d+", regex=True)
            is_dirty_numeric = has_currency | has_commas

            if is_dirty_numeric.any():
                bad_indices = sub_series[is_dirty_numeric].index
                bad_row_ids = df_with_id.loc[bad_indices, "_row_id"].tolist()
                sample_vals = series.loc[bad_indices].head(5).tolist()
                issues.append(Issue(
                    issue_id=f"FMT_NUM_{col}",
                    issue_type="invalid_format",
                    columns=[col],
                    row_ids=bad_row_ids,
                    sample_values=sample_vals,
                    severity="med",
                    description=f"Column '{col}' contains {len(bad_row_ids)} numeric values formatted as strings with currency symbols or commas.",
                    detector="format_detector",
                    evidence={
                        "format_type": "numeric_as_text",
                        "dirty_count": len(bad_row_ids),
                        "currency_symbol_count": int(has_currency.sum()),
                        "comma_count": int(has_commas.sum())
                    }
                ))

    return issues
