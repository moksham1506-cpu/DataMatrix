"""DataFix: Autonomous Data Quality and Repair Engine - Streamlit Application."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Optional
import pandas as pd
import streamlit as st

from datafix.config import Config
from datafix.pipeline import DataFixPipeline, PipelineResult
from datafix.profiler import profile_dataset
from datafix.ui_helpers import (
    validate_and_load_file,
    filter_issues,
    apply_change_decisions,
    apply_bulk_threshold,
    generate_export_bundle,
)

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DataFix - Autonomous Data Quality & Repair",
    page_icon="🛠️",
    layout="wide",
)

st.title("🛠️ DataFix: Autonomous Data Quality & Repair Engine")
st.caption("We don't just clean data, we prove what we changed, why, and how it was validated.")

# -----------------------------------------------------------------------------
# CACHED PIPELINE EXECUTION
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_demo_data_cached() -> pd.DataFrame:
    """Load the demo dataset from disk with caching."""
    demo_paths = [
        Path("data/demo_dataset.csv"),
        Path(__file__).parent / "data" / "demo_dataset.csv",
        Path("data/customers.csv"),
    ]
    for p in demo_paths:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError("Demo dataset not found in data/ directory.")


def run_pipeline(df: pd.DataFrame, dataset_name: str) -> PipelineResult:
    """Execute the DataFix pipeline on the dataset."""
    pipeline = DataFixPipeline()
    return pipeline.run(df, dataset_name=dataset_name)


# -----------------------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -----------------------------------------------------------------------------
if "df_raw" not in st.session_state:
    st.session_state.df_raw = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = ""
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "audit_log" not in st.session_state:
    st.session_state.audit_log = None
if "df_repaired" not in st.session_state:
    st.session_state.df_repaired = None
if "decisions" not in st.session_state:
    st.session_state.decisions = {}

# -----------------------------------------------------------------------------
# MAIN TABS SETUP
# -----------------------------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Upload & Profile",
    "2. Detected Issues",
    "3. Root-Cause Diagnosis",
    "4. Repair Review",
    "5. Validation",
    "6. Export",
])

# =============================================================================
# TAB 1: UPLOAD AND PROFILE
# =============================================================================
with tab1:
    st.header("Dataset Ingestion & Quality Profiling")
    st.markdown("Upload a CSV or Excel dataset to begin autonomous detection, diagnosis, and safe repair.")

    col_up, col_demo = st.columns([3, 1])
    with col_up:
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file",
            type=["csv", "xlsx", "xls"],
            help="Select a CSV or Excel file. DataFix will assign a stable `_row_id` without modifying original data."
        )
    with col_demo:
        st.write("")
        st.write("")
        load_demo_btn = st.button("🚀 Load Demo Dataset (5k rows)", use_container_width=True)

    # Process file upload or demo data trigger
    new_df = None
    new_name = None

    if uploaded_file is not None:
        loaded_df, err = validate_and_load_file(uploaded_file)
        if err:
            st.error(f"❌ {err}")
        else:
            new_df = loaded_df
            new_name = uploaded_file.name

    elif load_demo_btn:
        try:
            demo_raw = load_demo_data_cached()
            loaded_df, err = validate_and_load_file(demo_raw)
            if err:
                st.error(f"❌ {err}")
            else:
                new_df = loaded_df
                new_name = "demo_dataset.csv"
        except Exception as e:
            st.error(f"❌ Failed to load demo dataset: {e}")

    # If new dataset loaded, run pipeline and populate session state
    if new_df is not None and (st.session_state.df_raw is None or st.session_state.dataset_name != new_name):
        with st.spinner("Analyzing dataset with autonomous detectors and diagnosis engine..."):
            st.session_state.df_raw = new_df
            st.session_state.dataset_name = new_name
            res = run_pipeline(new_df, dataset_name=new_name)
            st.session_state.pipeline_result = res
            st.session_state.audit_log = res.audit_log
            st.session_state.df_repaired = res.repaired_df
            st.session_state.decisions = {e.change_id: e.status for e in res.audit_log.entries}
        st.success(f"Successfully loaded and analyzed **{new_name}** ({len(new_df):,} rows)")

    # Display profiling info if dataset is loaded
    if st.session_state.df_raw is not None and st.session_state.pipeline_result is not None:
        profile = st.session_state.pipeline_result.profile_before
        
        st.divider()
        st.subheader("Dataset Summary & Key Dimensions")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Rows", f"{profile.total_rows:,}")
        kpi2.metric("Total Columns", f"{profile.total_columns}")
        kpi3.metric("Memory Usage", f"{profile.memory_bytes / 1024:.1f} KB")
        kpi4.metric("Overall Missingness", f"{profile.overall_missing_pct:.2f}%")

        st.subheader("Column Profiles & Inferred Types")
        col_summary_data = []
        for col_name, cprof in profile.column_profiles.items():
            if col_name == "_row_id":
                continue
            col_summary_data.append({
                "Column": col_name,
                "Inferred Type": cprof.inferred_type,
                "Null Count": cprof.null_count,
                "Null %": f"{cprof.null_pct:.1f}%",
                "Distinct Values": cprof.distinct_count,
                "Sample Values": ", ".join(str(v) for v in cprof.sample_values[:3])
            })
        st.dataframe(pd.DataFrame(col_summary_data), use_container_width=True)

        with st.expander("Preview Raw Ingested Data (first 10 rows)", expanded=False):
            st.dataframe(st.session_state.df_raw.head(10), use_container_width=True)
    else:
        st.info("👆 Upload a dataset or click **'Load Demo Dataset'** to inspect profiles and run DataFix.")

# =============================================================================
# TAB 2: DETECTED ISSUES
# =============================================================================
with tab2:
    st.header("Detected Data Quality Issues")
    if st.session_state.pipeline_result is None:
        st.info("Please load a dataset in Tab 1 to view detected issues.")
    else:
        issues = st.session_state.pipeline_result.issues_before
        unique_types = sorted(list({i.issue_type for i in issues}))

        # Filters
        fcol1, fcol2, fcol3 = st.columns([2, 2, 2])
        with fcol1:
            type_filter = st.selectbox("Filter by Issue Type", options=["All"] + unique_types)
        with fcol2:
            sev_filter = st.selectbox("Filter by Severity", options=["All", "high", "med", "low"])
        with fcol3:
            st.write("")
            st.write("")
            st.caption(f"Showing issues matching selected filters")

        filtered_issues = filter_issues(issues, issue_type=type_filter, severity=sev_filter)

        # Issue Breakdown KPIs
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Issues Detected", len(issues))
        m2.metric("High Severity", sum(1 for i in issues if i.severity == "high"))
        m3.metric("Medium Severity", sum(1 for i in issues if i.severity == "med"))
        m4.metric("Low Severity", sum(1 for i in issues if i.severity == "low"))

        st.subheader(f"Issues List ({len(filtered_issues)} displayed)")
        if filtered_issues:
            issue_rows = []
            for i in filtered_issues:
                issue_rows.append({
                    "Issue ID": i.issue_id,
                    "Type": i.issue_type,
                    "Severity": i.severity.upper(),
                    "Columns": ", ".join(i.columns),
                    "Affected Rows": len(i.row_ids),
                    "Detector": i.detector,
                    "Description": i.description,
                    "Sample Values": ", ".join(str(v) for v in i.sample_values[:3])
                })
            st.dataframe(pd.DataFrame(issue_rows), use_container_width=True)

            with st.expander("🔍 Inspect Issue Details & Evidence", expanded=False):
                selected_issue_id = st.selectbox(
                    "Select an Issue ID to inspect full evidence",
                    options=[i.issue_id for i in filtered_issues]
                )
                selected_issue = next((i for i in filtered_issues if i.issue_id == selected_issue_id), None)
                if selected_issue:
                    st.write(f"**Description:** {selected_issue.description}")
                    st.write(f"**Columns:** {', '.join(selected_issue.columns)}")
                    st.write(f"**Detector:** `{selected_issue.detector}` | **Severity:** `{selected_issue.severity}`")
                    st.write(f"**Affected Row IDs ({len(selected_issue.row_ids)} total):** {selected_issue.row_ids[:20]}{' ...' if len(selected_issue.row_ids) > 20 else ''}")
                    st.json(selected_issue.evidence)
        else:
            st.info("No issues match the selected filter criteria.")

# =============================================================================
# TAB 3: ROOT-CAUSE DIAGNOSIS
# =============================================================================
with tab3:
    st.header("Root-Cause Diagnosis")
    st.markdown("The diagnostic engine clusters data quality defects into systemic root-cause patterns.")

    if st.session_state.pipeline_result is None:
        st.info("Please load a dataset in Tab 1 to view root causes.")
    else:
        diagnoses = st.session_state.pipeline_result.diagnoses
        if not diagnoses:
            st.info("No systemic root-cause patterns identified for the detected issues.")
        else:
            st.write(f"Identified **{len(diagnoses)} root-cause clusters** across detected issues:")

            for idx, diag in enumerate(diagnoses, 1):
                with st.container(border=True):
                    head_col1, head_col2 = st.columns([4, 1])
                    with head_col1:
                        st.subheader(f"Cluster #{idx}: {diag.root_cause_label}")
                    with head_col2:
                        conf_pct = int(diag.confidence * 100)
                        st.metric("Confidence", f"{conf_pct}%")

                    # Metadata row
                    c1, c2, c3 = st.columns(3)
                    c1.write(f"**Affected Columns:** `{', '.join(diag.affected_columns)}`")
                    c2.write(f"**Affected Rows:** `{diag.affected_row_count:,}`")
                    c3.write(f"**Linked Issues:** `{len(diag.issue_ids)}`")

                    # Explanation and Recommendation
                    if diag.explanation:
                        st.write(f"💡 **Diagnosis:** {diag.explanation}")
                    if diag.recommended_action:
                        st.write(f"🛠️ **Recommended Action:** {diag.recommended_action}")

                    # Evidence display
                    if diag.evidence:
                        with st.expander("🔬 Supporting Statistical Evidence", expanded=False):
                            st.json(diag.evidence)

# =============================================================================
# TAB 4: REPAIR REVIEW
# =============================================================================
with tab4:
    st.header("Repair Review & Audit Decisions")
    st.markdown("Inspect proposed data repairs. High-confidence repairs are applied automatically; lower confidence repairs are suggested for human review.")

    if st.session_state.pipeline_result is None or st.session_state.audit_log is None:
        st.info("Please load a dataset in Tab 1 to view repairs.")
    else:
        audit_log = st.session_state.audit_log
        entries = audit_log.entries

        if not entries:
            st.info("No modifications or suggestions recorded for this dataset.")
        else:
            # Status summary metrics
            counts = audit_log.summary()
            rc1, rc2, rc3, rc4 = st.columns(4)
            rc1.metric("Applied Automatically", counts.get("applied", 0))
            rc2.metric("Pending Suggestions", counts.get("suggested", 0))
            rc3.metric("User Approved", counts.get("approved", 0))
            rc4.metric("User Rejected", counts.get("rejected", 0))

            st.divider()

            # Bulk Action and Single-Change Action controls
            act_col1, act_col2 = st.columns([1, 2])
            with act_col1:
                st.subheader("⚡ Bulk Action")
                if st.button("Apply all above threshold (≥ 0.85)", use_container_width=True):
                    approved_count = apply_bulk_threshold(audit_log, threshold=0.85)
                    # Sync decisions
                    for e in audit_log.entries:
                        st.session_state.decisions[e.change_id] = e.status
                    # Recompute repaired DataFrame
                    df_repaired, audit_log = apply_change_decisions(
                        st.session_state.df_raw, audit_log, st.session_state.decisions
                    )
                    st.session_state.df_repaired = df_repaired
                    st.session_state.audit_log = audit_log
                    st.success(f"Applied {approved_count} changes above threshold!")
                    st.rerun()

            with act_col2:
                st.subheader("✍️ Approve / Reject Individual Change")
                change_options = [
                    f"{e.change_id} | Row {e.row_id} | Col: {e.column} | [{e.status.upper()}] (Conf: {e.confidence:.2f})"
                    for e in entries
                ]
                sel_change_str = st.selectbox("Select Change", options=change_options)
                sel_change_id = sel_change_str.split(" | ")[0] if sel_change_str else None

                btn_col1, btn_col2 = st.columns(2)
                with btn_col1:
                    if st.button("✅ Approve Selected", use_container_width=True):
                        st.session_state.decisions[sel_change_id] = "approved"
                        df_repaired, audit_log = apply_change_decisions(
                            st.session_state.df_raw, audit_log, st.session_state.decisions
                        )
                        st.session_state.df_repaired = df_repaired
                        st.session_state.audit_log = audit_log
                        st.success(f"Approved change {sel_change_id}")
                        st.rerun()
                with btn_col2:
                    if st.button("❌ Reject Selected", use_container_width=True):
                        st.session_state.decisions[sel_change_id] = "rejected"
                        df_repaired, audit_log = apply_change_decisions(
                            st.session_state.df_raw, audit_log, st.session_state.decisions
                        )
                        st.session_state.df_repaired = df_repaired
                        st.session_state.audit_log = audit_log
                        st.warning(f"Rejected change {sel_change_id}")
                        st.rerun()

            st.divider()

            # Table of Changes
            st.subheader("Audit Trail: Table of Changes")
            table_rows = []
            for e in entries:
                table_rows.append({
                    "change_id": e.change_id,
                    "row_id": e.row_id,
                    "column": e.column,
                    "old_value": str(e.old_value),
                    "new_value": str(e.new_value),
                    "confidence": round(e.confidence, 3),
                    "reason": e.reason,
                    "status": e.status
                })
            df_changes = pd.DataFrame(table_rows)
            st.dataframe(df_changes, use_container_width=True)

with tab5:
    st.header("Validation")
    st.info("Please load a dataset in Tab 1 to view validation results.")

with tab6:
    st.header("Export")
    st.info("Please load a dataset in Tab 1 to export results.")
