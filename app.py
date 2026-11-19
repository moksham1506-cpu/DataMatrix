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
from datafix.validate import validate_repair
from datafix.ui_helpers import (
    validate_and_load_file,
    filter_issues,
    apply_change_decisions,
    apply_bulk_threshold,
    compute_projected_threshold_impact,
    generate_export_bundle,
    paginate_entries,
    build_diff_dataframe,
    style_confidence_dataframe,
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
st.caption("We don't just clean data, we prove what we changed.")

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


@st.cache_data(show_spinner=False)
def load_benchmark_scoreboard_cached():
    """Retrieve ground-truth scoreboard metrics with caching."""
    from benchmark import get_benchmark_scoreboard
    return get_benchmark_scoreboard()


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
if "initial_decisions" not in st.session_state:
    st.session_state.initial_decisions = {}

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
            st.session_state.initial_decisions = {e.change_id: e.status for e in res.audit_log.entries}
        st.success(f"Successfully loaded and analyzed **{new_name}** ({len(new_df):,} rows)")

    # Display profiling info if dataset is loaded
    if st.session_state.df_raw is not None and st.session_state.pipeline_result is not None:
        profile = st.session_state.pipeline_result.profile_before
        
        st.divider()
        st.subheader("Dataset Summary & Key Dimensions")
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)
        kpi1.metric("Total Rows", f"{profile.row_count:,}")
        kpi2.metric("Total Columns", f"{profile.column_count}")
        kpi3.metric("Memory Usage", f"{profile.memory_bytes / 1024:.1f} KB")
        total_cells = max(profile.row_count * profile.column_count, 1)
        total_nulls = sum(cp.null_count for cp in profile.column_profiles.values())
        kpi4.metric("Overall Missingness", f"{(total_nulls / total_cells) * 100:.2f}%")

        st.subheader("Column Profiles & Inferred Types")
        col_summary_data = []
        for col_name, cprof in profile.column_profiles.items():
            if col_name == "_row_id":
                continue
            col_summary_data.append({
                "Column": col_name,
                "Inferred Type": cprof.inferred_type,
                "Null Count": cprof.null_count,
                "Null %": f"{cprof.null_ratio * 100:.1f}%",
                "Distinct Values": cprof.unique_count,
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

            # What-If Threshold Simulator
            st.subheader("🎚️ Threshold What-If Simulator")
            st.markdown("Move the auto-apply threshold to preview live defect resolution and projected data quality score without re-running detection.")
            
            whatif_col_slider, whatif_col_btn = st.columns([3, 1])
            with whatif_col_slider:
                slider_val = st.slider(
                    "Auto-Apply Confidence Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.85,
                    step=0.05,
                    help="Repairs with confidence ≥ this threshold are projected as applied."
                )
            
            impact = compute_projected_threshold_impact(
                df_original=st.session_state.df_raw,
                issues_before=st.session_state.pipeline_result.issues_before,
                audit_log=audit_log,
                threshold=slider_val
            )

            with whatif_col_btn:
                st.write("")
                st.write("")
                if st.button(f"⚡ Apply Threshold ({slider_val:.2f})", use_container_width=True):
                    approved_count = apply_bulk_threshold(audit_log, threshold=slider_val)
                    for e in audit_log.entries:
                        st.session_state.decisions[e.change_id] = e.status
                    df_repaired, audit_log = apply_change_decisions(
                        st.session_state.df_raw, audit_log, st.session_state.decisions
                    )
                    st.session_state.df_repaired = df_repaired
                    st.session_state.audit_log = audit_log
                    st.success(f"Applied threshold {slider_val:.2f}: {approved_count} changes updated!")
                    st.rerun()

            sc1, sc2, sc3, sc4 = st.columns(4)
            sc1.metric("Projected Quality Score", f"{impact['projected_score']:.1f} / 100", delta=f"{impact['score_delta']:+.1f}")
            sc2.metric("Projected Applied Changes", f"{impact['applied_count']:,}")
            sc3.metric("Projected Suggested Changes", f"{impact['suggested_count']:,}")
            sc4.metric("Projected Resolved Issues", f"{impact['resolved_issues_count']} / {len(st.session_state.pipeline_result.issues_before)}")

            st.divider()

            # Bulk Action and Single-Change Action controls
            act_col1, act_col2 = st.columns([1, 2])
            with act_col1:
                st.subheader("⚡ Bulk Action")
                if st.button("Apply all above threshold (≥ 0.70)", use_container_width=True):
                    approved_count = apply_bulk_threshold(audit_log, threshold=0.70)
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
                st.subheader("✍️ Approve / Reject / Undo Changes")
                change_options = [
                    f"{e.change_id} | Row {e.row_id} | Col: {e.column} | [{e.status.upper()}] (Conf: {e.confidence:.2f})"
                    for e in entries
                ]
                sel_change_str = st.selectbox("Select Change", options=change_options)
                sel_change_id = sel_change_str.split(" | ")[0] if sel_change_str else None

                btn_col1, btn_col2, btn_col3 = st.columns(3)
                with btn_col1:
                    if st.button("✅ Approve", use_container_width=True):
                        st.session_state.decisions[sel_change_id] = "approved"
                        df_repaired, audit_log = apply_change_decisions(
                            st.session_state.df_raw, audit_log, st.session_state.decisions
                        )
                        st.session_state.df_repaired = df_repaired
                        st.session_state.audit_log = audit_log
                        st.success(f"Approved change {sel_change_id}")
                        st.rerun()
                with btn_col2:
                    if st.button("❌ Reject", use_container_width=True):
                        st.session_state.decisions[sel_change_id] = "rejected"
                        df_repaired, audit_log = apply_change_decisions(
                            st.session_state.df_raw, audit_log, st.session_state.decisions
                        )
                        st.session_state.df_repaired = df_repaired
                        st.session_state.audit_log = audit_log
                        st.warning(f"Rejected change {sel_change_id}")
                        st.rerun()
                with btn_col3:
                    if st.button("↩️ Undo", use_container_width=True):
                        orig_status = st.session_state.initial_decisions.get(sel_change_id, "suggested")
                        st.session_state.decisions[sel_change_id] = orig_status
                        df_repaired, audit_log = apply_change_decisions(
                            st.session_state.df_raw, audit_log, st.session_state.decisions
                        )
                        st.session_state.df_repaired = df_repaired
                        st.session_state.audit_log = audit_log
                        st.info(f"Reverted {sel_change_id} to {orig_status}")
                        st.rerun()

            st.divider()

            # Cell-Level Diff Viewer with Confidence Color Coding and Pagination
            st.subheader("🔍 Cell-Level Diff Viewer (Color-coded by Confidence)")
            
            # Confidence Color Legend
            leg_c1, leg_c2, leg_c3 = st.columns(3)
            leg_c1.markdown("🟢 **High Confidence (≥ 0.85)**: Auto-applied safe repairs", help="Green background")
            leg_c2.markdown("🟡 **Medium Confidence (0.60–0.85)**: Suggestions for human review", help="Amber background")
            leg_c3.markdown("🔴 **Low Confidence (< 0.60)**: High discretion repairs", help="Red background")

            # Pagination controls (200 rows per page)
            page_size = 200
            total_pages = max(1, (len(entries) + page_size - 1) // page_size)

            pcol1, pcol2 = st.columns([1, 3])
            with pcol1:
                current_page = st.number_input(
                    f"Page (1 to {total_pages})",
                    min_value=1,
                    max_value=total_pages,
                    value=1,
                    step=1
                )
            with pcol2:
                start_row = (current_page - 1) * page_size + 1
                end_row = min(current_page * page_size, len(entries))
                st.write("")
                st.caption(f"Displaying modifications **{start_row:,}** to **{end_row:,}** of **{len(entries):,}** total changes")

            page_entries, _ = paginate_entries(entries, page=current_page, page_size=page_size)
            df_diff = build_diff_dataframe(page_entries)
            
            # Render styled dataframe with cell-level colors and old -> new diff
            styled_diff = style_confidence_dataframe(df_diff)
            st.dataframe(styled_diff, use_container_width=True)

# =============================================================================
# TAB 5: VALIDATION
# =============================================================================
with tab5:
    st.header("Repair Validation & Quality Score Impact")
    st.markdown("Post-repair verification executes full re-detection to prove issue resolution and ensure zero unintended defects.")

    if (st.session_state.df_raw is None or 
        st.session_state.df_repaired is None or 
        st.session_state.pipeline_result is None):
        st.info("Please load a dataset in Tab 1 to view validation results.")
    else:
        # Recompute fresh validation on the current state of repairs
        validation = validate_repair(
            df_original=st.session_state.df_raw,
            df_repaired=st.session_state.df_repaired,
            issues_before=st.session_state.pipeline_result.issues_before
        )

        mb = validation.metrics_before
        ma = validation.metrics_after

        # Top-level Quality Score Cards
        qcol1, qcol2, qcol3 = st.columns(3)
        qcol1.metric("Data Quality Score (Before)", f"{mb.overall_score:.1f} / 100")
        qcol2.metric("Data Quality Score (After)", f"{ma.overall_score:.1f} / 100", delta=f"{validation.score_delta:+.1f}")
        qcol3.metric("Newly Introduced Issues", f"{validation.issues_newly_introduced_count}", delta="0 defects added" if validation.issues_newly_introduced_count == 0 else "WARNING")

        st.divider()

        # Dimension Breakdown Table
        st.subheader("Data Quality Dimensions: Before vs After")
        dim_data = [
            {
                "Dimension": "Overall Quality Score",
                "Weight": "100%",
                "Score Before": f"{mb.overall_score:.1f}",
                "Score After": f"{ma.overall_score:.1f}",
                "Delta": f"{ma.overall_score - mb.overall_score:+.1f}"
            },
            {
                "Dimension": "Completeness",
                "Weight": "30%",
                "Score Before": f"{mb.completeness:.1f}",
                "Score After": f"{ma.completeness:.1f}",
                "Delta": f"{ma.completeness - mb.completeness:+.1f}"
            },
            {
                "Dimension": "Validity",
                "Weight": "30%",
                "Score Before": f"{mb.validity:.1f}",
                "Score After": f"{ma.validity:.1f}",
                "Delta": f"{ma.validity - mb.validity:+.1f}"
            },
            {
                "Dimension": "Uniqueness",
                "Weight": "20%",
                "Score Before": f"{mb.uniqueness:.1f}",
                "Score After": f"{ma.uniqueness:.1f}",
                "Delta": f"{ma.uniqueness - mb.uniqueness:+.1f}"
            },
            {
                "Dimension": "Consistency",
                "Weight": "20%",
                "Score Before": f"{mb.consistency:.1f}",
                "Score After": f"{ma.consistency:.1f}",
                "Delta": f"{ma.consistency - mb.consistency:+.1f}"
            },
        ]
        st.dataframe(pd.DataFrame(dim_data), use_container_width=True)

        # ---------------------------------------------------------------------
        # Collateral Damage & Key Integrity Assessment (Feature 5)
        # ---------------------------------------------------------------------
        if validation.collateral_damage:
            cd = validation.collateral_damage
            st.divider()
            st.subheader("🛡️ Collateral Damage & Key Integrity Assessment")
            st.markdown("Proves that repairs remained targeted: verifies that originally valid cells were untouched and keys were preserved.")

            col_cd1, col_cd2, col_cd3, col_cd4 = st.columns(4)
            col_cd1.metric(
                "Valid Cells Unchanged",
                f"{cd.valid_cells_unchanged_pct:.2f}%",
                delta="Target: 100%"
            )
            col_cd2.metric(
                "Unintended Changes",
                f"{cd.unintended_changes_count}",
                delta="0 edits" if cd.unintended_changes_count == 0 else f"{cd.unintended_changes_count} unintended"
            )
            col_cd3.metric(
                "Clean Rows Affected",
                f"{cd.changed_cells_in_clean_rows}",
                delta="Target: 0 clean rows"
            )
            row_delta = cd.row_count_after - cd.row_count_before
            col_cd4.metric(
                "Dataset Row Count",
                f"{cd.row_count_after:,}",
                delta=f"{row_delta:+,} rows ({cd.rows_removed} dups dropped)" if cd.rows_removed > 0 else "No row loss"
            )

            # Key Integrity Report
            pk_fk = cd.pk_fk_integrity
            if pk_fk.get("integrity_preserved", True):
                st.success("✅ **Primary & Foreign Key Integrity 100% Preserved**: Zero orphaned foreign keys or duplicate primary keys introduced.")
            else:
                st.warning("⚠️ **Key Integrity Warning**: Discrepancies detected in relational key constraints.")

            if pk_fk.get("primary_keys"):
                with st.expander("🔑 Inspect Primary Key Columns Integrity", expanded=False):
                    pk_rows = []
                    for col, stat in pk_fk["primary_keys"].items():
                        pk_rows.append({
                            "Key Column": col,
                            "Unique Before": stat.get("unique_before"),
                            "Unique After": stat.get("unique_after"),
                            "Nulls Before": stat.get("nulls_before"),
                            "Nulls After": stat.get("nulls_after"),
                            "Integrity Preserved": "✅ Yes" if stat.get("preserved") else "❌ No"
                        })
                    st.dataframe(pd.DataFrame(pk_rows), use_container_width=True)

        st.divider()

        # Defect Resolution Summary
        st.subheader("Defect Resolution Summary")
        rcol1, rcol2, rcol3 = st.columns(3)
        rcol1.metric("Total Issues (Initial)", validation.issues_before_count)
        rcol2.metric("Issues Successfully Resolved", validation.issues_resolved_count)
        rcol3.metric("Issues Remaining", validation.issues_remaining_count)

        # Unresolved Issues Table
        st.subheader("Unresolved Issues Table")
        if validation.remaining_issues:
            unres_rows = []
            for issue in validation.remaining_issues:
                reason = validation.unresolved_reasons.get(
                    issue.issue_id,
                    "Ambiguous repair or confidence below auto_apply_threshold"
                )
                unres_rows.append({
                    "Issue ID": issue.issue_id,
                    "Type": issue.issue_type,
                    "Columns": ", ".join(issue.columns),
                    "Affected Rows": len(issue.row_ids),
                    "Severity": issue.severity.upper(),
                    "Reason Unresolved": reason
                })
            st.dataframe(pd.DataFrame(unres_rows), use_container_width=True)
        else:
            st.success("🎉 All detected quality issues have been completely resolved!")

        # ---------------------------------------------------------------------
        # Ground-Truth Scoreboard (Feature 6: Demo Dataset Only)
        # ---------------------------------------------------------------------
        is_demo = any(k in str(st.session_state.dataset_name).lower() for k in ["demo", "customer", "order"])
        if is_demo:
            st.divider()
            st.subheader("🏆 Ground-Truth Benchmark Scoreboard (Demo Dataset)")
            st.markdown("Precision, recall, and repair accuracy evaluated against known injected errors in `ground_truth.json`:")

            try:
                board_data = load_benchmark_scoreboard_cached()
                bsummary = board_data["summary"]

                bcol1, bcol2, bcol3, bcol4 = st.columns(4)
                bcol1.metric("Detection Recall", f"{bsummary['overall_detection_recall']:.1f}%", help="Percentage of true injected defects detected")
                bcol2.metric("Detection Precision", f"{bsummary['overall_detection_precision']:.1f}%", help="Percentage of reported issues that were actual defects")
                bcol3.metric("Repair Accuracy", f"{bsummary['repair_accuracy']:.1f}%", help="Percentage of repaired cells matching ground truth exactly")
                bcol4.metric("Unintended Changes", f"{bsummary.get('unintended_changes', 0)}", delta="0 edits outside ground truth" if bsummary.get('unintended_changes', 0) == 0 else "WARNING")

                cat_rows = []
                for itype, m in bsummary.get("metrics_by_issue_type", {}).items():
                    cat_rows.append({
                        "Issue Category": itype.replace("_", " ").title(),
                        "Ground Truth Count": m["ground_truth_count"],
                        "Detected Count": m["detected_count"],
                        "Precision": f"{m['precision']:.1f}%",
                        "Recall": f"{m['recall']:.1f}%",
                        "F1-Score": f"{m['f1_score']:.1f}%"
                    })
                st.dataframe(pd.DataFrame(cat_rows), use_container_width=True)
            except Exception as e:
                st.error(f"Failed to calculate benchmark scoreboard: {e}")

# =============================================================================
# TAB 6: EXPORT
# =============================================================================
with tab6:
    st.header("Export Cleaned Data & Verification Artifacts")
    st.markdown("Download verified datasets and complete audit artifacts. All exports strictly reflect your approval and rejection decisions.")

    if (st.session_state.df_raw is None or 
        st.session_state.df_repaired is None or 
        st.session_state.audit_log is None or
        st.session_state.pipeline_result is None):
        st.info("Please load a dataset in Tab 1 to export results.")
    else:
        # Generate export bundle reflecting user's approvals and rejections
        clean_name = (st.session_state.dataset_name or "dataset").replace(".csv", "").replace(".xlsx", "").replace(".xls", "").lower()
        
        bundle = generate_export_bundle(
            original_df=st.session_state.df_raw,
            repaired_df=st.session_state.df_repaired,
            audit_log=st.session_state.audit_log,
            issues_before=st.session_state.pipeline_result.issues_before,
            diagnoses=st.session_state.pipeline_result.diagnoses,
            dataset_name=clean_name
        )

        st.subheader("📦 Download Verification Package")
        dcol1, dcol2 = st.columns(2)
        with dcol1:
            st.download_button(
                label="📥 Download Cleaned Dataset (CSV)",
                data=bundle["cleaned_csv"],
                file_name=f"cleaned_{clean_name}.csv",
                mime="text/csv",
                use_container_width=True
            )
            st.download_button(
                label="📋 Download Change Log (CSV)",
                data=bundle["change_log_csv"],
                file_name=f"change_log_{clean_name}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with dcol2:
            st.download_button(
                label="🌐 Download HTML Quality Report",
                data=bundle["quality_report_html"],
                file_name=f"quality_report_{clean_name}.html",
                mime="text/html",
                use_container_width=True
            )
            st.download_button(
                label="🧾 Download Change Log (JSON)",
                data=bundle["change_log_json"],
                file_name=f"change_log_{clean_name}.json",
                mime="application/json",
                use_container_width=True
            )

        st.success("✅ Exports are synchronized with your review decisions.")

        with st.expander("📄 Preview Audit Trail JSON", expanded=False):
            st.code(bundle["change_log_json"][:2000] + ("\n... [truncated]" if len(bundle["change_log_json"]) > 2000 else ""), language="json")
