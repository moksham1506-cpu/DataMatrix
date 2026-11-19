"""DataFix Pipeline Orchestrator: End-to-end Detect -> Diagnose -> Repair -> Validate workflow."""

from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
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

    def export(self, output_dir: str = "outputs", base_name: Optional[str] = None) -> Dict[str, str]:
        """Export cleaned dataset, HTML quality report, and audit logs (CSV and JSON)."""
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

        return {
            "cleaned_csv": str(cleaned_csv_path),
            "change_log_csv": str(audit_csv_path),
            "change_log_json": str(audit_json_path),
            "quality_report_html": str(report_html_path),
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
            validation=validation
        )


def run_pipeline(
    df: pd.DataFrame,
    ref_dfs: Optional[Dict[str, pd.DataFrame]] = None,
    config: Optional[Config] = None,
    dataset_name: str = "Dataset"
) -> PipelineResult:
    """Convenience helper to run the DataFix pipeline."""
    pipeline = DataFixPipeline(config=config)
    return pipeline.run(df, ref_dfs=ref_dfs, dataset_name=dataset_name)
