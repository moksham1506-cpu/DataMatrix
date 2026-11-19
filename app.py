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

# Placeholders for remaining tabs
with tab2:
    st.header("Detected Issues")
    st.info("Please load a dataset in Tab 1 to view detected issues.")

with tab3:
    st.header("Root-Cause Diagnosis")
    st.info("Please load a dataset in Tab 1 to view root causes.")

with tab4:
    st.header("Repair Review")
    st.info("Please load a dataset in Tab 1 to view repairs.")

with tab5:
    st.header("Validation")
    st.info("Please load a dataset in Tab 1 to view validation results.")

with tab6:
    st.header("Export")
    st.info("Please load a dataset in Tab 1 to export results.")
