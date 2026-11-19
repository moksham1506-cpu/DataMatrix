"""DataFix Performance Benchmark: Evaluates throughput and timing per stage on 100k rows."""

import sys
import time
from pathlib import Path

# Ensure root repository directory is on sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from datafix.config import Config
from datafix.profiler import ensure_row_id, profile_dataset
from datafix.detectors import run_all_detectors
from datafix.diagnose import diagnose_root_causes
from datafix.repair import repair_dataset
from datafix.validate import validate_repair


def run_100k_benchmark(n_rows: int = 100000):
    print(f"Generating synthetic {n_rows:,}-row dataset...")
    np.random.seed(42)

    df_100k = pd.DataFrame({
        "order_id": np.arange(100000, 100000 + n_rows),
        "customer_id": np.random.randint(1000, 9999, size=n_rows),
        "category": np.random.choice(["Electronics", "electronics", "ELECTRONICS", "Home", "Health", "Home & Garden"], size=n_rows),
        "quantity": np.random.randint(1, 10, size=n_rows),
        "unit_price": np.round(np.random.uniform(5.0, 500.0, size=n_rows), 2),
        "total_amount": np.zeros(n_rows, dtype=object),
        "order_date": np.random.choice(["2023-01-15", "15/01/2023", "2023.05.20", "2023-08-10"], size=n_rows),
        "status": np.random.choice(["Shipped", "shipped", "SHIPPED", "delivered", "DELIVERED", "pending"], size=n_rows),
        "country": np.random.choice(["USA", "United States", "us", "U.S.A.", "UK", "United Kingdom", "Canada"], size=n_rows),
        "age": np.random.choice([25.0, 35.0, 45.0, -5.0, 55.0, 120.0], size=n_rows)
    })

    qty = df_100k["quantity"].values
    price = df_100k["unit_price"].values
    totals = np.round(qty * price, 2)
    totals_list = [str(t) for t in totals]

    for i in range(0, n_rows, 500):
        totals_list[i] = f"${totals[i]}"
    for i in range(1, n_rows, 1000):
        totals_list[i] = f"USD {totals[i]}"
    for i in range(2, n_rows, 2000):
        totals_list[i] = str(round(totals[i] * 0.5, 2))

    df_100k["total_amount"] = totals_list

    mem_mb = df_100k.memory_usage().sum() / (1024 * 1024)
    print(f"Dataset generated: {n_rows:,} rows x {len(df_100k.columns)} columns ({mem_mb:.1f} MB)")

    cfg = Config(seed=42)

    # Stage 0: Row ID Ingestion
    t0 = time.perf_counter()
    df_id = ensure_row_id(df_100k)
    stage0_time = time.perf_counter() - t0

    # Stage 1: Profiling Before
    t0 = time.perf_counter()
    prof_before = profile_dataset(df_id, config=cfg)
    stage1_time = time.perf_counter() - t0

    # Stage 2: Defect Detection
    t0 = time.perf_counter()
    issues = run_all_detectors(df_id, config=cfg)
    stage2_time = time.perf_counter() - t0

    # Stage 3: Root-Cause Diagnosis
    t0 = time.perf_counter()
    diagnoses = diagnose_root_causes(issues, df_id, config=cfg)
    stage3_time = time.perf_counter() - t0

    # Stage 4: Conservative Repair
    t0 = time.perf_counter()
    df_rep, audit = repair_dataset(df_id, issues, config=cfg)
    stage4_time = time.perf_counter() - t0

    # Stage 5: Profiling After
    t0 = time.perf_counter()
    prof_after = profile_dataset(df_rep, config=cfg)
    stage5_time = time.perf_counter() - t0

    # Stage 6: Validation & Collateral Damage
    t0 = time.perf_counter()
    val = validate_repair(df_id, df_rep, issues, config=cfg)
    stage6_time = time.perf_counter() - t0

    total_time = stage0_time + stage1_time + stage2_time + stage3_time + stage4_time + stage5_time + stage6_time
    throughput = n_rows / total_time

    print("\n" + "=" * 65)
    print("      DATAFIX PERFORMANCE BENCHMARK (100,000 ROWS)")
    print("=" * 65)
    print(f"  Stage 0: Row ID Ingestion         : {stage0_time:6.3f} s  ({stage0_time/total_time*100:4.1f}%)")
    print(f"  Stage 1: Pre-Repair Profiling     : {stage1_time:6.3f} s  ({stage1_time/total_time*100:4.1f}%)")
    print(f"  Stage 2: Defect Detection         : {stage2_time:6.3f} s  ({stage2_time/total_time*100:4.1f}%)")
    print(f"  Stage 3: Root-Cause Diagnosis     : {stage3_time:6.3f} s  ({stage3_time/total_time*100:4.1f}%)")
    print(f"  Stage 4: Conservative Repair      : {stage4_time:6.3f} s  ({stage4_time/total_time*100:4.1f}%)")
    print(f"  Stage 5: Post-Repair Profiling    : {stage5_time:6.3f} s  ({stage5_time/total_time*100:4.1f}%)")
    print(f"  Stage 6: Validation & Collateral  : {stage6_time:6.3f} s  ({stage6_time/total_time*100:4.1f}%)")
    print("-" * 65)
    print(f"  Total End-to-End Execution Time   : {total_time:6.3f} s")
    print(f"  Overall Processing Throughput     : {throughput:8,.0f} rows/second")
    print("-" * 65)
    print(f"  Total Quality Issues Detected     : {len(issues):,}")
    print(f"  Total Repairs Applied             : {len(audit.filter_by_status('applied')):,}")
    print(f"  Collateral Damage / Unintended    : {val.collateral_damage.unintended_changes_count}")
    print(f"  Valid Cells Unchanged             : {val.collateral_damage.valid_cells_unchanged_pct:.2f}%")
    print(f"  Quality Score Delta               : +{val.score_delta:.2f} ({val.metrics_before.overall_score:.1f} -> {val.metrics_after.overall_score:.1f})")
    print("=" * 65 + "\n")

    return {
        "stage0_time": round(stage0_time, 3),
        "stage1_time": round(stage1_time, 3),
        "stage2_time": round(stage2_time, 3),
        "stage3_time": round(stage3_time, 3),
        "stage4_time": round(stage4_time, 3),
        "stage5_time": round(stage5_time, 3),
        "stage6_time": round(stage6_time, 3),
        "total_time": round(total_time, 3),
        "throughput": round(throughput, 0),
    }


if __name__ == "__main__":
    run_100k_benchmark()
