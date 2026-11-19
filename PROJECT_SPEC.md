# DataFix: Autonomous Data Quality and Repair Engine

## 1. PROJECT GOAL
Problem statement: Design a working prototype that accepts a dataset (CSV/Excel), DETECTS quality issues (missing values, duplicates, invalid formats, inconsistent entries, anomalies, broken relationships), DIAGNOSES likely root causes (not just fixed rules), REPAIRS safely, VALIDATES the result, EXPLAINS every change with a confidence score, and TRACKS everything in an auditable log with before/after values. It must export: cleaned dataset, quality report, modification log.
Core principle: "We don't just clean data, we prove what we changed, why, and how it was validated."
Judging criteria: detection accuracy and consistency, safe repairs with minimal unintended changes, explainability and confidence scoring, scalability, usability, reproducibility.

## 2. HARD CONSTRAINTS
- Python 3.11+, pandas, numpy, scikit-learn, rapidfuzz, streamlit, plotly, jinja2, pytest. No other heavy dependencies. Pin versions in requirements.txt.
- The core pipeline must run fully offline with NO API keys. An optional LLM explanation feature may exist behind a config flag (default OFF) and must fall back to template-based explanations.
- Never modify the original dataframe. Assign a stable `_row_id` to every row on load and keep it through all steps.
- Deterministic and reproducible: a single seed in config, same input + config = same output.
- Safe repairs only: every repair gets a confidence score in [0,1]. Repairs >= auto_apply_threshold (default 0.85) are applied automatically; repairs below it are only SUGGESTED and need user approval. Never silently change valid values. Never merge/delete rows if it would break key or foreign-key relationships.
- Idempotent: running the pipeline on its own output must produce zero new changes.
- Handle at least 100k rows without crashing (vectorized pandas, no row-by-row loops where avoidable).

## 3. REPO STRUCTURE
The repo root is DataMatrix (already created on GitHub). The Python package inside it is named `datafix`.
DataMatrix/
  app.py                      # Streamlit UI
  datafix/
    __init__.py
    config.py                 # thresholds, seed, weights
    models.py                 # Issue and ChangeLogEntry dataclasses
    profiler.py               # column type inference + stats
    detectors/                # missing.py, duplicates.py, formats.py, consistency.py, outliers.py, relationships.py
    diagnose.py               # groups issues into root-cause patterns
    repair.py                 # repair strategies per issue type
    confidence.py             # confidence scoring
    validate.py               # re-detect, quality score before/after
    audit.py                  # change log, undo/approve/reject
    report.py                 # HTML quality report (jinja2)
    pipeline.py               # orchestrates detect -> diagnose -> repair -> validate
    cli.py                    # python -m datafix.cli input.csv --out outputs/
  data/generate_demo_data.py  # creates messy demo CSVs + ground_truth.json of injected errors
  tests/
  docs/
  README.md, requirements.txt, .gitignore, PROJECT_SPEC.md (copy of this prompt)

## 4. DATA MODELS (implement exactly)
Issue: issue_id, issue_type, columns, row_ids, sample_values, severity (low/med/high), description, detector, evidence (dict).
ChangeLogEntry: change_id, timestamp, row_id, column, old_value, new_value, issue_id, strategy, reason (human-readable), confidence, status (applied | suggested | approved | rejected | reverted).

## 5. CAPABILITIES
DETECT: missing values (incl. disguised: "N/A", "null", "-", "?"), exact and fuzzy duplicates (rapidfuzz, similarity threshold in config), invalid formats (dates, emails, phones, numerics stored as text), inconsistent categories ("USA" vs "U.S.A." vs "United States"; case/whitespace variants), outliers (IQR/z-score + IsolationForest for multivariate), relationship violations (orphan foreign keys, cross-column rules like end_date < start_date, negative ages, totals not matching parts).
DIAGNOSE: cluster errors by pattern and infer likely cause (e.g. mixed date formats from two sources, unit mismatch, column shift, copy-paste typo, disguised nulls, encoding issues). Output a root-cause label plus supporting evidence for each cluster. Statistical/pattern based; LLM optional.
REPAIR (conservative): type-aware imputation (median/mode, group-wise where a group column exists, otherwise flag only), standardize formats, map category variants to canonical form, merge duplicates only above a high similarity threshold and only if keys stay consistent, fix cross-column violations only when unambiguous. Anything ambiguous becomes a suggestion with an explanation, never an auto-change.
VALIDATE: re-run all detectors on the repaired data; report issues resolved, remaining, and newly introduced (must be zero); compute a Data Quality Score (0-100) from weighted completeness, validity, uniqueness, consistency; show before vs after; list unresolved issues with reasons.
EXPLAIN + TRACK: every change has reason + confidence + before/after value; support approve/reject/undo per change and re-export after reviewing.

## 6. STREAMLIT UI (app.py)
Tabs: 1 Upload and Profile | 2 Detected Issues (filterable by type/severity) | 3 Root-Cause Diagnosis | 4 Repair Review (table of changes with confidence, approve/reject toggles, bulk "apply all above threshold") | 5 Validation (before/after quality score, charts, unresolved issues) | 6 Export (cleaned CSV, quality_report.html, change_log.csv + change_log.json). Include a "Load demo dataset" button. Clean, simple layout; use st.cache_data for heavy steps.

## 7. DEMO DATA AND BENCHMARK
data/generate_demo_data.py must create a realistic dataset (e.g. customers/orders, ~5k rows, two related tables) with KNOWN injected errors across every issue type and save ground_truth.json. Add a benchmark script that reports detection precision/recall per issue type and repair accuracy against ground truth. Put results in the README.

## 8. GIT AND GITHUB WORKFLOW (MANDATORY, DO IT CONTINUOUSLY)
- Remote: https://github.com/moksham1506-cpu/DataMatrix.git. Team: code blooded. Branch: main.
- First thing: git init (if needed), run `git remote add origin https://github.com/moksham1506-cpu/DataMatrix.git`, set the branch to main, create .gitignore (venv, __pycache__, .env, outputs/, *.pyc, .DS_Store), make the first commit, push with `git push -u origin main`.
- After EVERY meaningful unit of work (each module + its tests passing) run: git add -A, git commit with a conventional message (feat:, fix:, test:, docs:, chore:), then git push origin main. Never batch a whole phase into one commit.
- Never commit secrets, .env files, API keys, or large data files (>5 MB). Never force push. Never rewrite history.
- If a push fails (auth, conflict), STOP, show me the exact error, and wait. Do not try workarounds.
- Keep the repo runnable at every commit: `pip install -r requirements.txt && streamlit run app.py` must work.

## 9. QUOTA-SAVING RULES (I have one account and 24 hours)
- Make small targeted edits. Do not regenerate whole files or re-read the entire repo each step.
- Do not add libraries or features not listed here. Do not refactor working code unless a test fails.
- Run tests with `pytest -q`, show only a short summary. Keep your explanations to a few lines.
- Write the code and tests together per module; fix failures immediately before moving on.

## 10. PHASES (stop after each and wait for me to say "continue")
Phase 0: repo setup, git + GitHub push, requirements, folder skeleton, models.py, config.py, PROJECT_SPEC.md.
Phase 1: demo data generator + ground truth, profiler.py, tests.
Phase 2: all detectors + tests.
Phase 3: diagnose.py, repair.py, confidence.py, audit.py + tests (idempotency and "original not mutated" tests required).
Phase 4: validate.py, report.py, pipeline.py, cli.py, benchmark script.
Phase 5: Streamlit UI (all six tabs) + exports.
Phase 6: polish, error handling for bad uploads (empty file, bad encoding, single column), README (overview, architecture diagram in mermaid, setup, usage, benchmark results, limitations), final push.
