"""DataFix UI Helper Functions: Non-UI business logic for Streamlit application."""

from __future__ import annotations
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from datafix.config import Config
from datafix.models import Issue, ChangeLogEntry
from datafix.profiler import ensure_row_id, safe_assign
from datafix.audit import AuditLog
from datafix.validate import validate_repair, ValidationResult, compute_quality_metrics
from datafix.report import generate_html_report

__all__ = [
    "validate_and_load_file",
    "filter_issues",
    "apply_change_decisions",
    "apply_bulk_threshold",
    "compute_projected_threshold_impact",
    "paginate_entries",
    "build_diff_dataframe",
    "style_confidence_dataframe",
    "generate_export_bundle",
    "create_quality_gauge",
    "create_column_heatmap",
]


def validate_and_load_file(uploaded_file: Any) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Validate and read an uploaded file or path into a pandas DataFrame.
    
    Supports CSV and Excel files. Performs safety validations:
    - Empty file detection
    - Encoding error detection with fallback
    - Single-column file detection (e.g., bad delimiter)
    
    Returns:
        (df, None) on success, or (None, error_message) on failure.
    """
    if uploaded_file is None:
        return None, "No file provided."

    # Handle DataFrame directly
    if isinstance(uploaded_file, pd.DataFrame):
        df = uploaded_file.copy()
        if df.empty or len(df) == 0:
            return None, "The dataset contains 0 rows. Please upload a dataset with data rows."
        if len(df.columns) <= 1:
            return None, "Dataset contains only a single column. Please verify delimiter or file structure."
        return ensure_row_id(df), None

    # Determine filename
    filename = ""
    if hasattr(uploaded_file, "name"):
        filename = str(uploaded_file.name).lower()
    elif isinstance(uploaded_file, (str, Path)):
        filename = str(uploaded_file).lower()

    # Check for empty file if size attribute is present
    if hasattr(uploaded_file, "size") and uploaded_file.size == 0:
        return None, "The uploaded file is empty (0 bytes). Please upload a valid CSV or Excel dataset."

    # Read based on extension
    df: Optional[pd.DataFrame] = None
    try:
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            df = pd.read_excel(uploaded_file)
        else:
            # CSV file: handle possible encodings
            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            
            # Read first chunk or entire buffer to check for 0 bytes
            encodings = ["utf-8", "latin1", "cp1252", "iso-8859-1"]
            parsed = False
            last_err = None

            for enc in encodings:
                try:
                    if hasattr(uploaded_file, "seek"):
                        uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=enc)
                    parsed = True
                    break
                except UnicodeDecodeError as ue:
                    last_err = ue
                    continue
                except Exception as e:
                    last_err = e
                    # If it's EmptyDataError
                    if "No columns to parse from file" in str(e) or "empty" in str(e).lower():
                        return None, "The uploaded file is empty or contains no readable data."
                    break

            if not parsed or df is None:
                return None, f"Unable to decode CSV file: {last_err}. Please ensure valid UTF-8 or standard encoding."

    except Exception as e:
        if "empty" in str(e).lower() or "no columns" in str(e).lower():
            return None, "The uploaded file is empty or contains no columns."
        return None, f"Failed to load dataset: {str(e)}"

    # Post-read validations
    if df is None or df.empty or len(df) == 0:
        return None, "The dataset contains 0 rows. Please upload a dataset with data rows."

    if len(df.columns) <= 1:
        return None, "Dataset contains only a single column. Please verify delimiter or file structure."

    # Assign stable row IDs
    df_with_id = ensure_row_id(df)
    return df_with_id, None


def filter_issues(
    issues: List[Issue],
    issue_type: Optional[str] = None,
    severity: Optional[str] = None
) -> List[Issue]:
    """Filter issues by issue_type and severity."""
    filtered = issues
    if issue_type and issue_type != "All":
        filtered = [i for i in filtered if i.issue_type == issue_type]
    if severity and severity != "All":
        filtered = [i for i in filtered if i.severity == severity]
    return filtered


def apply_change_decisions(
    original_df: pd.DataFrame,
    audit_log: AuditLog,
    decisions: Dict[str, str]
) -> Tuple[pd.DataFrame, AuditLog]:
    """Reconstruct repaired DataFrame based on user approval/rejection decisions.
    
    decisions maps change_id -> 'approved' | 'rejected' | 'applied' | 'suggested'
    """
    # Update statuses in audit_log
    for entry in audit_log.entries:
        if entry.change_id in decisions:
            entry.status = decisions[entry.change_id]

    df_repaired = ensure_row_id(original_df).copy()
    rows_to_drop = set()

    for entry in audit_log.entries:
        if entry.status in {"applied", "approved"}:
            if entry.column == "ALL" or str(entry.new_value) == "[Removed]":
                rows_to_drop.add(entry.row_id)
            elif entry.column in df_repaired.columns:
                row_mask = df_repaired["_row_id"] == entry.row_id
                if row_mask.any():
                    safe_assign(df_repaired, row_mask, entry.column, entry.new_value)

    if rows_to_drop:
        df_repaired = df_repaired[~df_repaired["_row_id"].isin(rows_to_drop)].copy()

    return df_repaired, audit_log


def apply_bulk_threshold(
    audit_log: AuditLog,
    threshold: float = 0.85
) -> int:
    """Mark all suggested or rejected changes with confidence >= threshold as approved.
    
    Returns number of newly approved entries.
    """
    count = 0
    for entry in audit_log.entries:
        if entry.status in {"suggested", "rejected"} and entry.confidence >= threshold:
            entry.status = "approved"
            count += 1
    return count


def compute_projected_threshold_impact(
    df_original: pd.DataFrame,
    issues_before: List[Issue],
    audit_log: AuditLog,
    threshold: float,
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """Calculate the live impact of a threshold on applied/suggested counts and quality score without re-running detection."""
    cfg = config or Config()
    applied_count = sum(1 for e in audit_log.entries if e.confidence >= threshold)
    suggested_count = sum(1 for e in audit_log.entries if e.confidence < threshold)

    # Group change entries by issue_id
    issue_entries: Dict[str, List[ChangeLogEntry]] = {}
    for e in audit_log.entries:
        if e.issue_id:
            issue_entries.setdefault(e.issue_id, []).append(e)

    # An issue is resolved if all of its repairs meet or exceed the threshold
    remaining_issues: List[Issue] = []
    for issue in issues_before:
        entries = issue_entries.get(issue.issue_id, [])
        if not entries:
            remaining_issues.append(issue)
        elif not all(e.confidence >= threshold for e in entries):
            remaining_issues.append(issue)

    # Quality metrics before
    metrics_before = compute_quality_metrics(df_original, issues_before, config=cfg)
    # Projected quality metrics with remaining issues
    projected_metrics = compute_quality_metrics(df_original, remaining_issues, config=cfg)
    score_delta = round(projected_metrics.overall_score - metrics_before.overall_score, 1)

    return {
        "applied_count": applied_count,
        "suggested_count": suggested_count,
        "projected_score": round(projected_metrics.overall_score, 1),
        "score_delta": score_delta,
        "metrics_before": metrics_before,
        "projected_metrics": projected_metrics,
        "resolved_issues_count": len(issues_before) - len(remaining_issues),
        "remaining_issues_count": len(remaining_issues),
    }


def paginate_entries(
    entries: List[ChangeLogEntry],
    page: int = 1,
    page_size: int = 200
) -> Tuple[List[ChangeLogEntry], int]:
    """Slice entries for the requested page and return (page_entries, total_pages)."""
    if not entries:
        return [], 1
    total_pages = max(1, (len(entries) + page_size - 1) // page_size)
    current_page = max(1, min(page, total_pages))
    start_idx = (current_page - 1) * page_size
    end_idx = min(start_idx + page_size, len(entries))
    return entries[start_idx:end_idx], total_pages


def build_diff_dataframe(entries: List[ChangeLogEntry]) -> pd.DataFrame:
    """Build a rich DataFrame of modifications showing old -> new, reason, and confidence."""
    rows = []
    for e in entries:
        conf = round(float(e.confidence), 3)
        if conf >= 0.85:
            tier = "High (≥0.85)"
        elif conf >= 0.60:
            tier = "Medium (0.60–0.85)"
        else:
            tier = "Low (<0.60)"

        rows.append({
            "change_id": e.change_id,
            "row_id": e.row_id,
            "column": e.column,
            "diff": f"{e.old_value}  ➔  {e.new_value}",
            "old_value": str(e.old_value),
            "new_value": str(e.new_value),
            "confidence": conf,
            "confidence_tier": tier,
            "reason": e.reason,
            "status": e.status,
        })
    if not rows:
        return pd.DataFrame(columns=[
            "change_id", "row_id", "column", "diff", "old_value", "new_value",
            "confidence", "confidence_tier", "reason", "status"
        ])
    return pd.DataFrame(rows)


def style_confidence_dataframe(df: pd.DataFrame) -> Any:
    """Colorize rows by confidence score: green (>=0.85), amber (0.60-0.85), red (<0.60)."""
    def _row_styler(row):
        conf = row.get("confidence", 0.0)
        try:
            cval = float(conf)
        except (ValueError, TypeError):
            cval = 0.0

        if cval >= 0.85:
            # Soft green
            bg = "background-color: #dcfce7; color: #166534;"
        elif cval >= 0.60:
            # Soft amber
            bg = "background-color: #fef3c7; color: #92400e;"
        else:
            # Soft red
            bg = "background-color: #fee2e2; color: #991b1b;"
        return [bg] * len(row)

    return df.style.apply(_row_styler, axis=1)


def generate_export_bundle(
    original_df: pd.DataFrame,
    repaired_df: pd.DataFrame,
    audit_log: AuditLog,
    issues_before: List[Issue],
    diagnoses: list,
    dataset_name: str = "dataset",
    config: Optional[Config] = None
) -> Dict[str, Any]:
    """Generate in-memory export data for download buttons."""
    cfg = config or Config()
    
    # 1. Cleaned CSV (drop _row_id)
    export_df = repaired_df.drop(columns=["_row_id"], errors="ignore")
    cleaned_csv_str = export_df.to_csv(index=False)
    
    # 2. Audit log CSV and JSON
    audit_csv_str = audit_log.to_dataframe().to_csv(index=False)
    audit_json_str = audit_log.to_json(indent=2)
    
    # 3. Fresh validation
    validation = validate_repair(
        df_original=original_df,
        df_repaired=repaired_df,
        issues_before=issues_before,
        config=cfg
    )
    
    # 4. HTML report
    html_report = generate_html_report(
        validation_result=validation,
        diagnoses=diagnoses,
        audit_log=audit_log,
        dataset_name=dataset_name,
        output_path=None
    )
    
    return {
        "cleaned_csv": cleaned_csv_str,
        "change_log_csv": audit_csv_str,
        "change_log_json": audit_json_str,
        "quality_report_html": html_report,
        "validation": validation
    }


def create_quality_gauge(score_before: float, score_after: float) -> Any:
    """Create a dual-indicator Plotly gauge comparing quality scores before vs after repair."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "indicator"}, {"type": "indicator"}]],
        subplot_titles=["Initial Quality Score", "Repaired Quality Score"]
    )

    sb = round(float(score_before), 1)
    sa = round(float(score_after), 1)

    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=sb,
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#ef4444" if sb < 60 else "#f59e0b" if sb < 85 else "#10b981"},
            "steps": [
                {"range": [0, 60], "color": "#fee2e2"},
                {"range": [60, 85], "color": "#fef3c7"},
                {"range": [85, 100], "color": "#dcfce7"},
            ],
        }
    ), row=1, col=1)

    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=sa,
        delta={"reference": sb, "increasing": {"color": "#10b981"}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "#10b981"},
            "steps": [
                {"range": [0, 60], "color": "#fee2e2"},
                {"range": [60, 85], "color": "#fef3c7"},
                {"range": [85, 100], "color": "#dcfce7"},
            ],
        }
    ), row=1, col=2)

    fig.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_column_heatmap(issues_before: List[Issue], issues_after: List[Issue]) -> Any:
    """Create a Plotly heatmap of defects per column by issue type before vs after repair."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    cols_set = set()
    types_set = set()
    counts_before: Dict[Tuple[str, str], int] = {}
    counts_after: Dict[Tuple[str, str], int] = {}

    for iss in issues_before:
        types_set.add(iss.issue_type)
        for c in iss.columns:
            cols_set.add(c)
            counts_before[(c, iss.issue_type)] = counts_before.get((c, iss.issue_type), 0) + len(iss.row_ids)

    for iss in issues_after:
        types_set.add(iss.issue_type)
        for c in iss.columns:
            cols_set.add(c)
            counts_after[(c, iss.issue_type)] = counts_after.get((c, iss.issue_type), 0) + len(iss.row_ids)

    if not cols_set or not types_set:
        fig = go.Figure()
        fig.update_layout(title="No issues detected to display", height=240)
        return fig

    cols = sorted(list(cols_set))
    types = sorted(list(types_set))

    z_before = [[counts_before.get((c, t), 0) for c in cols] for t in types]
    z_after = [[counts_after.get((c, t), 0) for c in cols] for t in types]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Issues Before Repair", "Issues After Repair"],
        shared_yaxes=True
    )

    fig.add_trace(
        go.Heatmap(z=z_before, x=cols, y=types, colorscale="Reds", showscale=False),
        row=1, col=1
    )
    fig.add_trace(
        go.Heatmap(z=z_after, x=cols, y=types, colorscale="Greens", showscale=True),
        row=1, col=2
    )

    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
    return fig
