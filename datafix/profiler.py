"""DataFix Profiler: Column type inference and statistical dataset profiling."""

from __future__ import annotations
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pandas as pd

from datafix.config import Config, DEFAULT_DISGUISED_NULLS


@dataclass
class ColumnProfile:
    """Statistical and semantic profile of a single column."""
    name: str
    dtype: str
    inferred_type: str  # numeric, datetime, category, text, email, phone, boolean, identifier
    total_count: int
    null_count: int
    null_ratio: float
    disguised_null_count: int
    unique_count: int
    unique_ratio: float
    sample_values: List[Any] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DatasetProfile:
    """Summary profile of the entire dataset."""
    row_count: int
    column_count: int
    memory_bytes: int
    duplicate_rows_count: int
    column_profiles: Dict[str, ColumnProfile] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "memory_bytes": self.memory_bytes,
            "duplicate_rows_count": self.duplicate_rows_count,
            "column_profiles": {k: v.to_dict() for k, v in self.column_profiles.items()}
        }


# Regular expression patterns for semantic type inference
EMAIL_REGEX = re.compile(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$")
PHONE_REGEX = re.compile(r"^\+?[0-9\s\-().]{7,20}$")
NUMERIC_CLEAN_REGEX = re.compile(r"^[\s$€£¥,]*[-+]?[0-9]+(\.[0-9]+)?[\s%]*$")


def ensure_row_id(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a stable `_row_id` column without mutating the original DataFrame."""
    if "_row_id" in df.columns and df["_row_id"].is_unique:
        return df.copy()
    df_copy = df.copy()
    if "_row_id" in df_copy.columns:
        df_copy = df_copy.drop(columns=["_row_id"])
    df_copy.insert(0, "_row_id", np.arange(len(df_copy), dtype=np.int64))
    return df_copy


def infer_column_type(series: pd.Series, disguised_nulls: Set[str]) -> str:
    """Infer the high-level semantic type of a pandas Series."""
    # Drop real nulls and disguised nulls for type inspection
    clean = series.dropna()
    if clean.empty:
        return "text"

    # Filter disguised nulls from string series
    if clean.dtype == object or isinstance(clean.dtype, pd.StringDtype):
        str_series = clean.astype(str).str.strip()
        lower_series = str_series.str.lower()
        mask_valid = ~lower_series.isin(disguised_nulls) & (str_series != "")
        clean = str_series[mask_valid]
        if clean.empty:
            return "text"

    # 1. Boolean check
    if pd.api.types.is_bool_dtype(clean.dtype):
        return "boolean"
    if clean.dtype == object:
        unique_lower = set(clean.astype(str).str.lower().unique())
        if unique_lower.issubset({"true", "false", "t", "f", "1", "0", "yes", "no", "y", "n"}):
            return "boolean"

    # 2. Native Numeric check
    if pd.api.types.is_numeric_dtype(clean.dtype):
        # Check if it could be an identifier (e.g. id, index, strictly sequential integer)
        if pd.api.types.is_integer_dtype(clean.dtype):
            if series.name and str(series.name).lower().endswith(("_id", "id", "_key", "key")):
                return "identifier"
        return "numeric"

    # 3. Native Datetime check
    if pd.api.types.is_datetime64_any_dtype(clean.dtype):
        return "datetime"

    # Sample for performance on large series (vectorized regex on sample)
    sample_size = min(len(clean), 500)
    sample = clean.iloc[:sample_size].astype(str)

    # 4. Check Datetime string parseability before phone
    parsed_date_count = 0
    try:
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        parsed_date_count = parsed.notna().sum()
    except Exception:
        parsed_date_count = 0

    if parsed_date_count / sample_size >= 0.75:
        return "datetime"

    # 5. Check if string is formatted numeric (currency, commas, percentage)
    match_numeric_count = sample.map(lambda x: bool(NUMERIC_CLEAN_REGEX.match(x.strip()))).sum()
    if match_numeric_count / sample_size >= 0.80:
        return "numeric"

    # 6. Check Email
    match_email_count = sample.map(lambda x: bool(EMAIL_REGEX.match(x.strip()))).sum()
    if match_email_count / sample_size >= 0.70:
        return "email"

    # 7. Check Phone
    match_phone_count = sample.map(lambda x: bool(PHONE_REGEX.match(x.strip()))).sum()
    col_name_lower = str(series.name).lower() if series.name else ""
    is_phone_name = any(kw in col_name_lower for kw in ["phone", "mobile", "cell", "tel", "fax"])
    if match_phone_count / sample_size >= 0.70 and (is_phone_name or sample.str.contains(r"[\+\(\)]").any()):
        return "phone"

    # 8. Check Identifier / Primary key
    if series.name and str(series.name).lower().endswith(("_id", "id", "_key", "key", "_code")):
        if clean.nunique() / len(clean) >= 0.90:
            return "identifier"

    # 9. Categorical vs Free Text
    unique_ratio = clean.nunique() / len(clean)
    if clean.nunique() <= 50 or unique_ratio < 0.15:
        return "category"

    return "text"


def profile_column(series: pd.Series, config: Optional[Config] = None) -> ColumnProfile:
    """Generate a rich statistical and type profile for a single column."""
    cfg = config or Config()
    disguised_set = set(token.lower() for token in cfg.disguised_nulls)

    total_count = len(series)
    null_count = int(series.isna().sum())
    
    # Calculate disguised nulls for object/string columns
    disguised_null_count = 0
    if series.dtype == object or isinstance(series.dtype, pd.StringDtype):
        str_s = series.dropna().astype(str).str.strip().str.lower()
        disguised_null_count = int(str_s.isin(disguised_set).sum())

    effective_nulls = null_count + disguised_null_count
    null_ratio = float(effective_nulls / total_count) if total_count > 0 else 0.0

    inferred_type = infer_column_type(series, disguised_set)

    # Unique values (excluding nulls)
    clean_series = series.dropna()
    unique_count = int(clean_series.nunique())
    unique_ratio = float(unique_count / total_count) if total_count > 0 else 0.0

    # Sample representative values (up to 5 unique non-null values)
    samples = clean_series.drop_duplicates().head(5).tolist()

    stats: Dict[str, Any] = {}
    if inferred_type == "numeric":
        # Extract numeric series for stats
        num_s = pd.to_numeric(
            clean_series.astype(str).str.replace(r"[\$,€£¥\s]", "", regex=True),
            errors="coerce"
        ).dropna()
        if not num_s.empty:
            stats = {
                "min": float(num_s.min()),
                "max": float(num_s.max()),
                "mean": float(num_s.mean()),
                "std": float(num_s.std()) if len(num_s) > 1 else 0.0,
                "median": float(num_s.median()),
                "q25": float(num_s.quantile(0.25)),
                "q75": float(num_s.quantile(0.75)),
            }
    elif inferred_type == "datetime":
        dt_s = pd.to_datetime(clean_series, errors="coerce")
        dt_s = dt_s.dropna()
        if not dt_s.empty:
            stats = {
                "min_date": str(dt_s.min()),
                "max_date": str(dt_s.max()),
            }
    elif inferred_type in {"category", "text", "email", "phone"}:
        top_counts = clean_series.value_counts().head(5).to_dict()
        stats = {
            "top_categories": {str(k): int(v) for k, v in top_counts.items()},
            "avg_length": float(clean_series.astype(str).str.len().mean()) if not clean_series.empty else 0.0,
        }

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        inferred_type=inferred_type,
        total_count=total_count,
        null_count=null_count,
        null_ratio=round(null_ratio, 4),
        disguised_null_count=disguised_null_count,
        unique_count=unique_count,
        unique_ratio=round(unique_ratio, 4),
        sample_values=samples,
        stats=stats,
    )


def profile_dataset(df: pd.DataFrame, config: Optional[Config] = None) -> DatasetProfile:
    """Generate a complete dataset profile without mutating the input DataFrame."""
    df_with_id = ensure_row_id(df)
    row_count = len(df_with_id)
    cols_to_profile = [col for col in df_with_id.columns if col != "_row_id"]
    column_count = len(cols_to_profile)
    memory_bytes = int(df_with_id.memory_usage(deep=True).sum())

    # Calculate exact duplicate rows (excluding _row_id)
    duplicate_rows_count = int(df_with_id[cols_to_profile].duplicated().sum()) if column_count > 0 else 0

    col_profiles: Dict[str, ColumnProfile] = {}
    for col in cols_to_profile:
        col_profiles[col] = profile_column(df_with_id[col], config=config)

    return DatasetProfile(
        row_count=row_count,
        column_count=column_count,
        memory_bytes=memory_bytes,
        duplicate_rows_count=duplicate_rows_count,
        column_profiles=col_profiles,
    )
