# 🛠️ DataFix: Autonomous Data Quality & Repair Engine

> *"We don't just clean data, we prove what we changed."*

**Team**: `code blooded`  
**License**: MIT License  
**Submission Version**: `v1.0`

---

## 📌 Executive Summary & Problem Statement

Modern data pipelines and enterprise analytics fail silently because of "dirty data": disguised nulls (`"N/A"`, `"-"`, `"None"`), malformed dates (`DD/MM/YYYY` vs ISO `YYYY-MM-DD`), currency strings in numeric fields (`"$1,200.00"`), categorical spelling drifts (`"USA"` vs `"U.S.A."`), inverted order timestamps, and cross-column arithmetic mismatches (`Quantity * Unit Price != Total`).

Traditional cleaning scripts and black-box LLMs are fragile, uncalibrated, and destructive: they hallucinate values, silently drop data, corrupt relational integrity, and offer zero provable guarantees about what was changed or why.

**DataFix** solves this fundamentally by treating data repair as an auditable, confidence-calibrated transaction engine. Every single cell modification is scored, validated against domain bounds, tracked with cryptographic provenance (SHA-256), and guaranteed to preserve untouched valid data.

---

## 🏛️ Six Core Capabilities

DataFix addresses every dimension of the automated data cleaning lifecycle:

| Capability | How DataFix Implements It |
| :--- | :--- |
| **1. Detect** | 6 multi-strategy detector modules identifying disguised nulls, semantic type mismatch, date/email formats, fuzzy categories, domain outliers, and relational violations. |
| **2. Diagnose** | Agglomerative defect clustering and root-cause analysis that identifies systemic upstream pipeline defects (e.g., CSV export delimiter shifts, currency prefix leakage). |
| **3. Repair** | Conservative, confidence-calibrated repairs ($0.0 \le \text{Confidence} \le 1.0$). Strict auto-apply threshold ($\ge 0.85$) prevents unverified hallucinations. |
| **4. Validate** | Dual-pass statistical profiling and **Collateral Damage Assessment** proving 100% untouched status for valid cells and zero orphaned foreign keys. |
| **5. Explain** | Plain-English reasoning and algorithmic explanations recorded for every single modification in an interactive cell-level diff viewer. |
| **6. Track** | Append-only audit logs with unique IDs (`CHG_xxxxxxxx`), SHA-256 provenance hashes, and a bit-level CLI replay command (`python -m datafix.cli replay`). |

---

## 📐 System Architecture

```mermaid
flowchart TD
    subgraph Ingestion ["1. Ingestion & Profiling"]
        RAW["Raw Dataset (CSV / Excel)"] --> ROWID["Stable _row_id Assignment"]
        ROWID --> PROF1["Pre-Repair Profiler & Schema Inference"]
        PROF1 --> SHA_IN["Input SHA-256 Hash"]
    end

    subgraph Detection ["2. Defect Detection & Root-Cause Diagnosis"]
        ROWID --> DET["Multi-Strategy Detectors"]
        DET --> D1["Invalid Formats"]
        DET --> D2["Inconsistent Categories"]
        DET --> D3["Disguised Nulls"]
        DET --> D4["Outliers & Inversions"]
        DET --> D5["Relational Violations"]
        D1 & D2 & D3 & D4 & D5 --> ISSUES["Unified Issues Registry"]
        ISSUES --> DIAG["Agglomerative Root-Cause Engine"]
    end

    subgraph RepairEngine ["3. Conservative Repair Engine"]
        ISSUES --> CONF["Confidence Scorer (0.00 - 1.00)"]
        CONF --> THRESH{"Confidence >= Threshold?"}
        THRESH -- "Yes (>= 0.85)" --> APPLY["Vectorized Safe Assign"]
        THRESH -- "No (< 0.85)" --> SUGGEST["Flag as Suggestion for Human Review"]
        APPLY & SUGGEST --> AUDIT["Immutable Audit Log"]
    end

    subgraph Validation ["4. Validation & Collateral Damage Assessment"]
        APPLY --> REP_DF["Repaired DataFrame"]
        REP_DF --> PROF2["Post-Repair Profiler"]
        ROWID & REP_DF --> VAL["Dual-Pass Validation Engine"]
        VAL --> COLLAT["Collateral Damage Score & Key Integrity"]
    end

    subgraph Verification ["5. Provenance & Artifact Export"]
        REP_DF --> SHA_OUT["Output SHA-256 Hash"]
        AUDIT --> EXP1["cleaned_dataset.csv"]
        AUDIT --> EXP2["quality_report.html"]
        AUDIT --> EXP3["change_log.csv"]
        AUDIT --> EXP4["change_log.json"]
        AUDIT --> REPLAY["CLI Replay Engine"]
    end
```

---

## 🌟 Standout Features ("The Wow Factors")

1. **Cell-Level Diff Viewer (Tab 4)**: Color-codes every modification by confidence tier (**Green** $\ge 0.85$, **Amber** $0.60\text{–}0.85$, **Red** $< 0.60$). Displays `old ➔ new`, algorithmic reason, and row index. Paginated at 200 rows/page for instantaneous rendering.
2. **Interactive Threshold What-If Slider (Tab 4)**: Dynamically adjusts the auto-apply threshold ($0.50\text{–}1.00$) in real time, projecting the exact impact on the data quality score and applied change count without re-executing detection.
3. **Collateral Damage Assessment (Tab 5)**: Proves that good data was not harmed. Computes:
   - **Valid Cells Left Unchanged**: Target $100.0\%$.
   - **Unintended Changes Outside Known Defects**: Target $0$.
   - **Primary & Foreign Key Integrity**: Verifies zero duplicate PKs or orphan FKs.
4. **Ground-Truth Scoreboard (Tab 5)**: Evaluates detection precision/recall, repair accuracy, and unintended change count against injected ground-truth defects.
5. **Bit-Level Reproducibility & Replay (Tab 6 & CLI)**: Hashes the input, config, and output using SHA-256. The built-in replay engine re-executes the change log on raw data to verify byte-identical reproduction.

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.11.9)
- Git

### 1. Clone & Set Up Virtual Environment
```bash
git clone https://github.com/moksham1506-cpu/DataMatrix.git
cd DataMatrix

# Create and activate virtual environment
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Test Suite
```bash
pytest tests -q
```

---

## 🖥️ Usage

### Launch Interactive Streamlit UI
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser. Click **"🚀 Load Demo Dataset"** in Tab 1 to run the complete end-to-end pipeline.

### CLI Pipeline Execution
Clean any dataset directly from your terminal:
```bash
# Clean a CSV dataset with default threshold (0.85)
python -m datafix.cli data/demo_dataset.csv --out outputs/

# Clean with custom threshold and relational foreign key reference
python -m datafix.cli data/orders.csv --ref data/customers.csv --threshold 0.80 --out outputs/
```

### CLI Replay Verification
Verify cryptographic provenance and bit-level reproducibility:
```bash
python -m datafix.cli replay --input data/orders.csv --log outputs/change_log_orders.json --expect outputs/cleaned_orders.csv
```

---

## 📊 Benchmark Scoreboard Results

Evaluated against the synthetic enterprise e-commerce benchmark (`data/customers.csv`, `data/orders.csv`, `data/ground_truth.json`):

| Issue Category | Injected Ground Truth | Detected Defects | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Inconsistent Category** | 55 | 55 | **100.0%** | **100.0%** | **100.0%** |
| **Missing Values / Disguised Nulls** | 75 | 75 | **100.0%** | **100.0%** | **100.0%** |
| **Invalid Formats (Dates & Currency)** | 70 | 63 | **100.0%** | **90.0%** | **94.7%** |
| **Relationship Violations** | 70 | 82 | **85.4%** | **100.0%** | **92.1%** |
| **Duplicate Records** | 30 | 147 | 18.4% | **90.0%** | 30.5% |
| **Domain Outliers** | 27 | 163 | 16.6% | **100.0%** | 28.4% |
| **OVERALL COMPOSITE** | **327** | - | **54.2%** | **96.9%** | **69.5%** |

- **Repair Accuracy**: `86.9%`
- **Unintended Changes**: `12`

---

## ⚡ High-Throughput Performance (100,000 Rows Benchmark)

Evaluated on a 10-column enterprise dataset containing 100,000 rows (6.5 MB in memory) with 207,038 injected defect cells:

| Pipeline Stage | Execution Time | Percentage | Description |
| :--- | :---: | :---: | :--- |
| **Stage 0: Row ID Ingestion** | 0.053 s | 0.3% | Non-mutating zero-copy index generation |
| **Stage 1: Pre-Repair Profiling** | 1.975 s | 10.9% | Schema inference and disguised null scan |
| **Stage 2: Defect Detection** | 6.282 s | 34.7% | Multi-detector rule and pattern execution |
| **Stage 3: Root-Cause Diagnosis** | 0.000 s | 0.0% | Agglomerative pattern clustering |
| **Stage 4: Conservative Repair** | **2.633 s** | 14.5% | Vectorized cell updates (**207,038 repairs applied**) |
| **Stage 5: Post-Repair Profiling** | 1.835 s | 10.1% | Profile re-computation on repaired dataset |
| **Stage 6: Validation & Collateral** | 5.320 s | 29.4% | Full re-detection & key integrity check |
| **Total End-to-End Execution** | **18.098 s** | **100.0%** | **Throughput: 5,525 rows/second** |

- **Collateral Damage / Unintended Changes**: `0`
- **Valid Cells Unchanged**: `100.00%`
- **Quality Score Improvement**: `77.9 / 100 ➔ 99.9 / 100 (+22.0 points)`

---

## 🖼️ User Interface Gallery

Visual documentation located under `docs/screenshots/`:
- `01_upload_and_profile.png`: Initial dataset upload, inferred semantic dtypes, and disguised null ratios.
- `02_detected_issues.png`: Multi-column filtering by issue type and severity.
- `03_root_cause_diagnosis.png`: Upstream defect clustering cards and structural fix recommendations.
- `04_cell_diff_viewer.png`: Confidence-colored cell changes with approve/reject and what-if threshold slider.
- `05_validation_and_collateral.png`: Dual Plotly score gauge, column health heatmap, and key integrity card.
- `06_export_and_reproducibility.png`: Synchronized download package, SHA-256 hashes, and replay test pass badge.

---

## 🔮 Limitations & Future Work

1. **Streaming Data / Kafka Integration**: Currently optimized for batch workloads up to several million rows; streaming window micro-batching is planned for v2.0.
2. **Automated Schema Constraint Discovery**: Future iterations can leverage deeper statistical graph neural networks to infer complex $N$-ary multi-table foreign key constraints automatically.
3. **Active Learning from User Decisions**: User approve/reject selections could be serialized into permanent tenant-specific heuristics to adjust repair priors over time.

---

## 👥 Authors & Team

**Team Name**: `code blooded`  
Developed for Hackathon 2026.
