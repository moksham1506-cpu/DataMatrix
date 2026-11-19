"""DataFix Benchmark Evaluator: Evaluates detection precision/recall and repair accuracy against ground_truth.json."""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple
import pandas as pd

from datafix.config import Config
from datafix.pipeline import DataFixPipeline


def evaluate_benchmark(
    ground_truth_path: str = "data/ground_truth.json",
    customers_csv: str = "data/customers.csv",
    orders_csv: str = "data/orders.csv",
    output_dir: str = "outputs"
) -> Dict[str, Any]:
    """Calculate detection precision, recall, F1, and repair accuracy against injected errors."""
    gt_file = Path(ground_truth_path)
    if not gt_file.exists():
        raise FileNotFoundError(f"Ground truth file '{ground_truth_path}' not found.")

    with open(gt_file, "r", encoding="utf-8") as f:
        gt_data = json.load(f)

    injections: List[Dict[str, Any]] = gt_data["injections"]

    # Load datasets
    df_cust = pd.read_csv(customers_csv)
    df_ord = pd.read_csv(orders_csv)

    pipeline = DataFixPipeline(config=Config(seed=42))

    # Run pipeline on customers
    res_cust = pipeline.run(df_cust, dataset_name="customers")

    # Run pipeline on orders with customers reference
    res_ord = pipeline.run(df_ord, ref_dfs={"customers": df_cust}, dataset_name="orders")

    # Index detected issues by (table, row_id, issue_type)
    detected_set: Set[Tuple[str, int, str]] = set()

    for issue in res_cust.issues_before:
        for r_id in issue.row_ids:
            detected_set.add(("customers", r_id, issue.issue_type))

    for issue in res_ord.issues_before:
        for r_id in issue.row_ids:
            detected_set.add(("orders", r_id, issue.issue_type))

    # Group ground truth by issue_type
    gt_by_type: Dict[str, Set[Tuple[str, int]]] = {}
    for inj in injections:
        t = inj["table"]
        r = inj["row_id"]
        itype = inj["issue_type"]
        gt_by_type.setdefault(itype, set()).add((t, r))

    # Calculate Precision, Recall, F1 per issue type
    all_types = sorted(list(gt_by_type.keys()))
    metrics_by_type: Dict[str, Dict[str, float]] = {}

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for itype in all_types:
        expected_rows = gt_by_type[itype]
        detected_rows = {(t, r) for t, r, it in detected_set if it == itype}

        tp = len(expected_rows.intersection(detected_rows))
        fn = len(expected_rows - detected_rows)
        fp = len(detected_rows - expected_rows)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        total_tp += tp
        total_fp += fp
        total_fn += fn

        metrics_by_type[itype] = {
            "ground_truth_count": len(expected_rows),
            "detected_count": len(detected_rows),
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision * 100, 1),
            "recall": round(recall * 100, 1),
            "f1_score": round(f1 * 100, 1),
        }

    overall_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
    overall_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
    overall_f1 = (2 * overall_prec * overall_rec) / (overall_prec + overall_rec) if (overall_prec + overall_rec) > 0 else 0.0

    # Evaluate Repair Accuracy against ground truth clean values
    applied_changes_map: Dict[Tuple[str, int, str], Any] = {}
    for entry in res_cust.audit_log.filter_by_status("applied"):
        applied_changes_map[("customers", entry.row_id, entry.column)] = str(entry.new_value)
    for entry in res_ord.audit_log.filter_by_status("applied"):
        applied_changes_map[("orders", entry.row_id, entry.column)] = str(entry.new_value)

    repairable_injections = [
        inj for inj in injections
        if inj["issue_type"] in {"invalid_format", "inconsistent_category", "missing_value", "duplicate_row"}
    ]
    correct_repairs = 0
    total_eval_repairs = 0

    for inj in repairable_injections:
        key = (inj["table"], inj["row_id"], inj["column"])
        if key in applied_changes_map:
            total_eval_repairs += 1
            repaired_val = applied_changes_map[key]
            clean_val = str(inj["clean_value"])
            # Format comparison flexibility (e.g. 1450.0 vs 1450.00, or NaN)
            if clean_val.lower() == repaired_val.lower() or (clean_val in repaired_val):
                correct_repairs += 1
            elif inj["issue_type"] == "missing_value" and (repaired_val.lower() in {"nan", "none"} or pd.isna(repaired_val)):
                correct_repairs += 1

    repair_accuracy = (correct_repairs / total_eval_repairs * 100) if total_eval_repairs > 0 else 100.0

    # Evaluate unintended changes vs ground truth
    injected_cells: Set[Tuple[str, int, str]] = set()
    for inj in injections:
        injected_cells.add((inj["table"], inj["row_id"], inj["column"]))

    unintended_applied_changes = 0
    for (t, r, c), val in applied_changes_map.items():
        if (t, r, c) not in injected_cells and c != "ALL":
            unintended_applied_changes += 1

    benchmark_summary = {
        "overall_detection_precision": round(overall_prec * 100, 1),
        "overall_detection_recall": round(overall_rec * 100, 1),
        "overall_detection_f1": round(overall_f1 * 100, 1),
        "repair_accuracy": round(repair_accuracy, 1),
        "unintended_changes": unintended_applied_changes,
        "total_injections_evaluated": len(injections),
        "metrics_by_issue_type": metrics_by_type
    }

    # Save benchmark results
    out_p = Path(output_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    with open(out_p / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    return benchmark_summary


def format_scoreboard_markdown(summary: Dict[str, Any]) -> str:
    """Format benchmark metrics into a compact, beautiful Markdown scoreboard."""
    lines = [
        "| Issue Category | Ground Truth | Detected | Precision | Recall | F1-Score |",
        "|:---|:---:|:---:|:---:|:---:|:---:|"
    ]
    for itype, m in summary["metrics_by_issue_type"].items():
        name = itype.replace("_", " ").title()
        lines.append(
            f"| {name} | {m['ground_truth_count']} | {m['detected_count']} | "
            f"{m['precision']}% | {m['recall']}% | {m['f1_score']}% |"
        )
    lines.append("|:---|:---:|:---:|:---:|:---:|:---:|")
    lines.append(
        f"| **OVERALL COMPOSITE** | **{summary['total_injections_evaluated']}** | - | "
        f"**{summary['overall_detection_precision']}%** | **{summary['overall_detection_recall']}%** | "
        f"**{summary['overall_detection_f1']}%** |"
    )
    lines.append("")
    lines.append(f"**Repair Accuracy**: `{summary['repair_accuracy']}%` &bull; **Unintended Changes**: `{summary.get('unintended_changes', 0)}`")
    return "\n".join(lines)


def get_benchmark_scoreboard(
    ground_truth_path: str = "data/ground_truth.json",
    customers_csv: str = "data/customers.csv",
    orders_csv: str = "data/orders.csv"
) -> Dict[str, Any]:
    """Public API for Streamlit UI and external callers to fetch the benchmark scoreboard."""
    summary = evaluate_benchmark(
        ground_truth_path=ground_truth_path,
        customers_csv=customers_csv,
        orders_csv=orders_csv
    )
    md_table = format_scoreboard_markdown(summary)
    return {
        "summary": summary,
        "markdown_table": md_table
    }


def print_benchmark_table(results: Dict[str, Any]):
    """Print a clean Markdown table of the benchmark results."""
    print("\n# DataFix Autonomous Quality & Repair Benchmark Results\n")
    print(format_scoreboard_markdown(results))


if __name__ == "__main__":
    res = evaluate_benchmark()
    print_benchmark_table(res)
