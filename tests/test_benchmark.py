"""Tests for benchmark scoreboard functionality."""

from pathlib import Path
from benchmark import get_benchmark_scoreboard, format_scoreboard_markdown


def test_get_benchmark_scoreboard():
    # Run scoreboard against generated demo datasets
    scoreboard = get_benchmark_scoreboard(
        ground_truth_path="data/ground_truth.json",
        customers_csv="data/customers.csv",
        orders_csv="data/orders.csv"
    )
    assert "summary" in scoreboard
    assert "markdown_table" in scoreboard

    summary = scoreboard["summary"]
    assert summary["overall_detection_recall"] >= 90.0
    assert summary["repair_accuracy"] >= 80.0
    assert "unintended_changes" in summary

    # Verify markdown table format
    md = scoreboard["markdown_table"]
    assert "| Issue Category | Ground Truth | Detected |" in md
    assert "**Repair Accuracy**" in md
    assert "**Unintended Changes**" in md
