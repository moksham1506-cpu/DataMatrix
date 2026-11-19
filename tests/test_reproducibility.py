"""Tests for cryptographic reproducibility, SHA-256 provenance metadata, and CLI replay."""

from pathlib import Path
import json
import pandas as pd
import pytest

from datafix.config import Config
from datafix.pipeline import DataFixPipeline, replay_audit_log
from datafix.cli import main


def test_provenance_metadata_and_hashes(tmp_path: Path):
    df = pd.DataFrame({
        "country": ["USA", "United States", "Canada"],
        "price": ["$10.00", "$20.00", "$30.00"]
    })
    pipeline = DataFixPipeline(config=Config(seed=42))
    res = pipeline.run(df, dataset_name="hash_test")

    meta = res.metadata
    assert meta is not None
    assert len(meta.input_sha256) == 64
    assert len(meta.config_sha256) == 64
    assert len(meta.output_sha256) == 64

    # Verify export includes metadata_<name>.json
    exports = res.export(output_dir=str(tmp_path), base_name="hash_test")
    assert "metadata_json" in exports
    meta_path = Path(exports["metadata_json"])
    assert meta_path.exists()

    with open(meta_path, "r", encoding="utf-8") as f:
        meta_dict = json.load(f)
    assert meta_dict["output_sha256"] == meta.output_sha256


def test_replay_audit_log_verification(tmp_path: Path):
    # Create input dataset
    input_csv = tmp_path / "raw_input.csv"
    df = pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Alice"],
        "price": ["$10.50", "USD 20.00", "$30.00", "$10.50"],
        "country": ["USA", "Canada", "Germany", "USA"]
    })
    df.to_csv(input_csv, index=False)

    pipeline = DataFixPipeline(config=Config(seed=42))
    res = pipeline.run(df, dataset_name="replay_test")
    exports = res.export(output_dir=str(tmp_path), base_name="replay_test")

    cleaned_csv = exports["cleaned_csv"]
    log_json = exports["change_log_json"]

    # Replay audit log on raw input
    replay_res = replay_audit_log(
        input_file=str(input_csv),
        log_file=str(log_json),
        expect_file=str(cleaned_csv)
    )

    assert replay_res["matches"] is True
    assert replay_res["replayed_sha256"] == replay_res["expected_sha256"]
    assert replay_res["applied_change_count"] > 0


def test_cli_replay_command(tmp_path: Path, monkeypatch):
    input_csv = tmp_path / "cli_raw.csv"
    df = pd.DataFrame({
        "country": ["USA", "United States"],
        "price": ["$15.00", "$25.00"]
    })
    df.to_csv(input_csv, index=False)

    pipeline = DataFixPipeline(config=Config(seed=42))
    res = pipeline.run(df, dataset_name="cli_rep")
    exports = res.export(output_dir=str(tmp_path), base_name="cli_rep")

    # Run CLI replay
    import sys
    test_args = [
        "datafix.cli",
        "replay",
        "--input", str(input_csv),
        "--log", exports["change_log_json"],
        "--expect", exports["cleaned_csv"]
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
