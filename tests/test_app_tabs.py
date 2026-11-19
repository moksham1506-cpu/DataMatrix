"""End-to-end integration test verifying Streamlit app.py works across all six tabs."""

import pytest
from streamlit.testing.v1 import AppTest


def test_app_initial_state():
    """Verify app.py loads without crashing and displays initial instructions."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception
    assert len(at.tabs) == 6
    assert at.title[0].value == "🛠️ DataFix: Autonomous Data Quality & Repair Engine"


def test_app_load_demo_data_and_tabs():
    """Verify loading the demo dataset runs the pipeline and populates all 6 tabs."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()
    assert not at.exception

    # Click the "Load Demo Dataset" button (button index 0)
    assert len(at.button) >= 1
    # Find the demo button
    demo_btn = next((b for b in at.button if "Demo Dataset" in b.label), None)
    assert demo_btn is not None
    demo_btn.click().run()

    assert not at.exception

    # Tab 1: Check metrics and dataframes
    assert len(at.metric) >= 4  # Total Rows, Total Columns, Memory, Missingness
    # Verify success banner
    assert any("Successfully loaded and analyzed" in s.value for s in at.success)

    # Tab 2: Check issues are populated
    # There should be selectboxes for filters and an issues dataframe
    assert len(at.dataframe) >= 2

    # Tab 4: Verify bulk apply button and approve/reject buttons exist
    bulk_btn = next((b for b in at.button if "above threshold" in b.label), None)
    assert bulk_btn is not None
    approve_btn = next((b for b in at.button if "Approve" in b.label), None)
    assert approve_btn is not None

    # Tab 6: Verify session state has export-ready artifacts and app has no errors
    assert at.session_state.df_repaired is not None
    assert at.session_state.audit_log is not None
    assert len(at.session_state.decisions) > 0
    assert not at.exception


def test_app_bulk_apply_above_threshold():
    """Verify that clicking 'Apply all above threshold' updates decisions and repairs."""
    at = AppTest.from_file("app.py", default_timeout=30)
    at.run()

    # Load demo dataset
    demo_btn = next((b for b in at.button if "Demo Dataset" in b.label), None)
    demo_btn.click().run()
    assert not at.exception

    # Find bulk apply button
    bulk_btn = next((b for b in at.button if "above threshold" in b.label), None)
    assert bulk_btn is not None
    bulk_btn.click().run()
    assert not at.exception

    # Verify that approved status exists in decisions
    decisions = at.session_state.decisions
    assert any(status == "approved" for status in decisions.values())

