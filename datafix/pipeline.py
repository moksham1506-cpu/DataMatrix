"""DataFix Pipeline Orchestrator: End-to-end Detect -> Diagnose -> Repair -> Validate workflow."""

from __future__ import annotations
import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd

from datafix.config import Config
from datafix.models import Issue
from datafix.profiler import ensure_row_id, profile_dataset, DatasetProfile
from datafix.detectors import run_all_detectors
from datafix.diagnose import diagnose_root_causes, Diagnosis
from datafix.repair import repair_dataset
from datafix.audit import AuditLog
from datafix.validate import validate_repair, ValidationResult
from datafix.report import generate_html_report


def compute_sha256(data: bytes | str) -> str:
    """Calculate SHA-256 hash of bytes or string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


@dataclass
class RunMetadata:
    """Cryptographic provenance and reproducibility tracking."""
    input_sha256: str
    config_sha256: str
    output_sha256: str
    timestamp: str
    row_count_before: int
    row_count_after: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PipelineResult:
    """Encapsulates the complete artifacts of a DataFix pipeline run."""
    dataset_name: str
    original_df: pd.DataFrame
    repaired_df: pd.DataFrame
    profile_before: DatasetProfile
    profile_after: DatasetProfile
    issues_before: List[Issue]
    issues_after: List[Issue]
    diagnoses: List[Diagnosis]
    audit_log: AuditLog
    validation: ValidationResult
    metadata: Optional[RunMetadata] = None

    def export(self, output_dir: str = "outputs", base_name: Optional[str] = None) -> Dict[str, str]:
        """Export cleaned dataset, HTML quality report, audit logs, and run metadata."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        name = base_name or self.dataset_name.lower().replace(" ", "_")

        # 1. Cleaned CSV
        cleaned_csv_path = out_path / f"cleaned_{name}.csv"
        # Export without internal _row_id column
        export_df = self.repaired_df.drop(columns=["_row_id"], errors="ignore")
        export_df.to_csv(cleaned_csv_path, index=False)

        # 2. Audit Log CSV & JSON
        audit_csv_path = out_path / f"change_log_{name}.csv"
        audit_json_path = out_path / f"change_log_{name}.json"
        self.audit_log.to_dataframe().to_csv(audit_csv_path, index=False)
        with open(audit_json_path, "w", encoding="utf-8") as f:
            f.write(self.audit_log.to_json())

        # 3. HTML Quality Report
        report_html_path = out_path / f"quality_report_{name}.html"
        generate_html_report(
            validation_result=self.validation,
            diagnoses=self.diagnoses,
            audit_log=self.audit_log,
            dataset_name=self.dataset_name,
            output_path=str(report_html_path)
        )

        # 4. Provenance Metadata JSON
        meta_json_path = out_path / f"metadata_{name}.json"
        if self.metadata:
            with open(meta_json_path, "w", encoding="utf-8") as f:
                json.dump(self.metadata.to_dict(), f, indent=2)

        return {
            "cleaned_csv": str(cleaned_csv_path),
            "change_log_csv": str(audit_csv_path),
            "change_log_json": str(audit_json_path),
            "quality_report_html": str(report_html_path),
            "metadata_json": str(meta_json_path) if self.metadata else "",
        }


class DataFixPipeline:
    """Autonomous data quality and repair pipeline."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

    def run(
        self,
        df: pd.DataFrame,
        ref_dfs: Optional[Dict[str, pd.DataFrame]] = None,
        dataset_name: str = "Dataset"
    ) -> PipelineResult:
        """Execute full detect -> diagnose -> repair -> validate cycle."""
        # Step 0: Ensure stable row ID without mutating original DataFrame
        df_with_id = ensure_row_id(df)

        # Step 1: Profile before repairs
        profile_before = profile_dataset(df_with_id, config=self.config)

        # Step 2: Detect all data quality issues
        issues_before = run_all_detectors(df_with_id, ref_dfs=ref_dfs, config=self.config)

        # Step 3: Diagnose root causes
        diagnoses = diagnose_root_causes(issues_before, df_with_id, config=self.config)

        # Step 4: Repair safely with confidence scoring
        repaired_df, audit_log = repair_dataset(df_with_id, issues_before, config=self.config)

        # Step 5: Profile after repairs
        profile_after = profile_dataset(repaired_df, config=self.config)

        # Step 6: Validate result (re-detect & compute quality score delta)
        validation = validate_repair(
            df_original=df_with_id,
            df_repaired=repaired_df,
            issues_before=issues_before,
            ref_dfs=ref_dfs,
            config=self.config
        )

        # Step 7: Cryptographic Run Metadata
        input_csv_bytes = df_with_id.drop(columns=["_row_id"], errors="ignore").to_csv(index=False).encode("utf-8")
        config_bytes = json.dumps(self.config.to_dict(), sort_keys=True).encode("utf-8")
        output_csv_bytes = repaired_df.drop(columns=["_row_id"], errors="ignore").to_csv(index=False).encode("utf-8")

        run_metadata = RunMetadata(
            input_sha256=hashlib.sha256(input_csv_bytes).hexdigest(),
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            output_sha256=hashlib.sha256(output_csv_bytes).hexdigest(),
            timestamp=datetime.now(timezone.utc).isoformat(),
            row_count_before=len(df_with_id),
            row_count_after=len(repaired_df)
        )

        return PipelineResult(
            dataset_name=dataset_name,
            original_df=df_with_id,
            repaired_df=repaired_df,
            profile_before=profile_before,
            profile_after=profile_after,
            issues_before=issues_before,
            issues_after=validation.remaining_issues,
            diagnoses=diagnoses,
            audit_log=audit_log,
            validation=validation,
            metadata=run_metadata
        )


def replay_audit_log(
    input_file: str,
    log_file: str,
    expect_file: str
) -> Dict[str, Any]:
    """Re-applies the audit log to input_file and verifies output matches expect_file."""
    # 1. Load input
    inp_path = Path(input_file)
    if inp_path.suffix.lower() in {".xlsx", ".xls"}:
        df_input = pd.read_excel(inp_path)
    else:
        df_input = pd.read_csv(inp_path)

    # Ensure stable row id
    df_replayed = ensure_row_id(df_input).copy()

    # 2. Load change log
    with open(log_file, "r", encoding="utf-8") as f:
        log_entries = json.load(f)

    # 3. Apply changes with status == "applied"
    rows_to_drop: Set[int] = set()
    for entry in log_entries:
        if entry.get("status") == "applied":
            r_id = entry.get("row_id")
            col = entry.get("column")
            new_val = entry.get("new_value")
            strategy = entry.get("strategy")

            if col == "ALL" and (new_val == "[Removed]" or strategy == "deduplicate_exact_row"):
                rows_to_drop.add(r_id)
            elif col in df_replayed.columns:
                row_mask = df_replayed["_row_id"] == r_id
                if row_mask.any():
                    if new_val is None or (isinstance(new_val, str) and new_val.lower() in {"nan", "none"}):
                        df_replayed.loc[row_mask, col] = np.nan
                    else:
                        df_replayed.loc[row_mask, col] = new_val

    if rows_to_drop:
        df_replayed = df_replayed[~df_replayed["_row_id"].isin(rows_to_drop)].copy()

    # Drop internal row id for comparison
    df_replayed = df_replayed.drop(columns=["_row_id"], errors="ignore")

    # 4. Load expected output
    exp_path = Path(expect_file)
    if exp_path.suffix.lower() in {".xlsx", ".xls"}:
        df_expected = pd.read_excel(exp_path)
    else:
        df_expected = pd.read_csv(exp_path)

    df_expected = df_expected.drop(columns=["_row_id"], errors="ignore")

    # Compute SHA-256 on CSV serialization
    replayed_bytes = df_replayed.to_csv(index=False).encode("utf-8")
    expected_bytes = df_expected.to_csv(index=False).encode("utf-8")

    replayed_sha256 = hashlib.sha256(replayed_bytes).hexdigest()
    expected_sha256 = hashlib.sha256(expected_bytes).hexdigest()

    matches = (replayed_sha256 == expected_sha256)
    return {
        "matches": matches,
        "replayed_sha256": replayed_sha256,
        "expected_sha256": expected_sha256,
        "replayed_rows": len(df_replayed),
        "expected_rows": len(df_expected),
        "applied_change_count": sum(1 for e in log_entries if e.get("status") == "applied")
    }


def run_pipeline(
    df: pd.DataFrame,
    ref_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None,
    dataset_name: str = "Dataset"
) -> PipelineResult:
    """Convenience helper to run the DataFix pipeline."""
    pipeline = DataFixPipeline(config=config)
    return pipeline.run(df, ref_dfs=ref_dfs, dataset_name=dataset_name)
