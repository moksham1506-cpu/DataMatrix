"""DataFix Command Line Interface: Run autonomous data quality and repair from the terminal."""

from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
import pandas as pd

from datafix.config import Config
from datafix.pipeline import DataFixPipeline


def main():
    parser = argparse.ArgumentParser(
        prog="python -m datafix.cli",
        description="DataFix: Autonomous Data Quality and Repair Engine."
    )
    parser.add_argument("input_file", help="Path to input dataset (CSV or Excel)")
    parser.add_argument("--out", "-o", default="outputs", help="Output directory for cleaned data, logs, and report")
    parser.add_argument("--ref", help="Optional path to reference table (e.g. customers.csv for foreign key validation)")
    parser.add_argument("--threshold", type=float, default=0.85, help="Confidence threshold for auto-applying repairs (default: 0.85)")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic random seed (default: 42)")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress verbose console output")

    args = parser.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: Input file '{args.input_file}' does not exist.", file=sys.stderr)
        sys.exit(1)

    # Load dataset
    if not args.quiet:
        print(f"[*] Loading dataset from {args.input_file}...")
    try:
        if input_path.suffix.lower() in {".xlsx", ".xls"}:
            df = pd.read_excel(input_path)
        else:
            df = pd.read_csv(input_path)
    except Exception as e:
        print(f"Error loading {args.input_file}: {e}", file=sys.stderr)
        sys.exit(1)

    # Optional reference dataset
    ref_dfs = None
    if args.ref:
        ref_path = Path(args.ref)
        if ref_path.exists():
            ref_name = ref_path.stem
            ref_df = pd.read_csv(ref_path) if ref_path.suffix.lower() != ".xlsx" else pd.read_excel(ref_path)
            ref_dfs = {ref_name: ref_df}
            if not args.quiet:
                print(f"[*] Loaded reference table '{ref_name}' with {len(ref_df)} rows.")

    # Initialize configuration
    cfg = Config(
        seed=args.seed,
        auto_apply_threshold=args.threshold
    )
    pipeline = DataFixPipeline(config=cfg)

    dataset_name = input_path.stem
    if not args.quiet:
        print(f"[*] Executing DataFix pipeline on '{dataset_name}' ({len(df)} rows, {len(df.columns)} columns)...")

    result = pipeline.run(df, ref_dfs=ref_dfs, dataset_name=dataset_name)

    # Print summary metrics to terminal
    if not args.quiet:
        print("\n" + "=" * 65)
        print(" DATA QUALITY SCORE SUMMARY")
        print("=" * 65)
        print(f"  Overall Score Before : {result.validation.metrics_before.overall_score:.1f} / 100")
        print(f"  Overall Score After  : {result.validation.metrics_after.overall_score:.1f} / 100")
        delta = result.validation.score_delta
        print(f"  Net Improvement      : {'+' if delta >= 0 else ''}{delta:.1f} points")
        print("-" * 65)
        print(f"  Issues Detected      : {result.validation.issues_before_count}")
        print(f"  Issues Resolved      : {result.validation.issues_resolved_count}")
        print(f"  Remaining/Suggested  : {result.validation.issues_remaining_count}")
        print(f"  Newly Introduced     : {result.validation.issues_newly_introduced_count} (must be 0)")
        print("-" * 65)
        summary = result.audit_log.summary()
        print(f"  Repairs Applied      : {summary['applied']}")
        print(f"  Repairs Suggested    : {summary['suggested']}")
        print("=" * 65 + "\n")

    # Export outputs
    exported_files = result.export(output_dir=args.out, base_name=dataset_name)
    if not args.quiet:
        print(f"[+] Cleaned dataset exported to: {exported_files['cleaned_csv']}")
        print(f"[+] Audit log (CSV) exported to: {exported_files['change_log_csv']}")
        print(f"[+] Audit log (JSON) exported to: {exported_files['change_log_json']}")
        print(f"[+] Interactive HTML report to : {exported_files['quality_report_html']}")
        print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
