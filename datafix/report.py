"""DataFix Quality Report Generator: Produces a self-contained, auditable HTML report."""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from jinja2 import Template

from datafix.audit import AuditLog
from datafix.diagnose import Diagnosis
from datafix.validate import ValidationResult


HTML_REPORT_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>DataFix Quality & Audit Report - {{ dataset_name }}</title>
  <style>
    :root {
      --primary: #1e40af;
      --primary-light: #3b82f6;
      --bg: #f8fafc;
      --card-bg: #ffffff;
      --text: #0f172a;
      --text-muted: #64748b;
      --border: #e2e8f0;
      --success: #16a34a;
      --warning: #f59e0b;
      --danger: #dc2626;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      margin: 0;
      padding: 24px;
    }
    .container {
      max-width: 1200px;
      margin: 0 auto;
    }
    header {
      background: var(--card-bg);
      padding: 32px;
      border-radius: 12px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      border: 1px solid var(--border);
      margin-bottom: 24px;
    }
    h1 {
      margin: 0 0 8px 0;
      font-size: 28px;
      color: var(--primary);
    }
    .badge {
      display: inline-block;
      padding: 4px 10px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }
    .badge-success { background: #dcfce7; color: var(--success); }
    .badge-warning { background: #fef3c7; color: var(--warning); }
    .badge-danger { background: #fee2e2; color: var(--danger); }
    .badge-info { background: #e0f2fe; color: #0284c7; }
    
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .card {
      background: var(--card-bg);
      padding: 20px;
      border-radius: 12px;
      border: 1px solid var(--border);
      box-shadow: 0 1px 2px rgba(0,0,0,0.04);
    }
    .card-title {
      font-size: 13px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .card-value {
      font-size: 32px;
      font-weight: 700;
      margin-bottom: 4px;
    }
    .card-subtext {
      font-size: 13px;
      color: var(--text-muted);
    }
    .score-delta-pos { color: var(--success); }
    .score-delta-neg { color: var(--danger); }
    
    .section {
      background: var(--card-bg);
      padding: 24px;
      border-radius: 12px;
      border: 1px solid var(--border);
      margin-bottom: 24px;
    }
    h2 {
      font-size: 18px;
      margin-top: 0;
      margin-bottom: 16px;
      border-bottom: 1px solid var(--border);
      padding-bottom: 8px;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
      margin-top: 12px;
    }
    th, td {
      padding: 10px 14px;
      text-align: left;
      border-bottom: 1px solid var(--border);
    }
    th {
      background: #f1f5f9;
      font-weight: 600;
      color: var(--text-muted);
    }
    tr:hover {
      background: #f8fafc;
    }
    .code {
      font-family: ui-monospace, Menlo, Monaco, "Cascadia Mono", monospace;
      font-size: 12px;
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div style="display: flex; justify-content: space-between; align-items: flex-start;">
        <div>
          <h1>DataFix Quality & Audit Report</h1>
          <p style="margin: 0; color: var(--text-muted);">
            Dataset: <strong>{{ dataset_name }}</strong> &bull; Generated: {{ timestamp }}
          </p>
        </div>
        <div>
          {% if validation.is_valid %}
          <span class="badge badge-success">&#10003; Verification Passed</span>
          {% else %}
          <span class="badge badge-danger">&#9888; Verification Review Needed</span>
          {% endif %}
        </div>
      </div>
      <p style="margin-top: 16px; margin-bottom: 0; font-size: 14px; font-style: italic; color: #475569;">
        "We don't just clean data, we prove what we changed, why, and how it was validated."
      </p>
    </header>

    <!-- Executive Metrics Grid -->
    <div class="grid">
      <div class="card">
        <div class="card-title">Data Quality Score</div>
        <div class="card-value">{{ validation.metrics_after.overall_score }} <span style="font-size: 18px; color: var(--text-muted);">/ 100</span></div>
        <div class="card-subtext">
          Before: {{ validation.metrics_before.overall_score }} 
          (<strong class="score-delta-pos">+{{ validation.score_delta }} pts</strong>)
        </div>
      </div>
      <div class="card">
        <div class="card-title">Completeness</div>
        <div class="card-value">{{ validation.metrics_after.completeness }}%</div>
        <div class="card-subtext">Before: {{ validation.metrics_before.completeness }}%</div>
      </div>
      <div class="card">
        <div class="card-title">Validity</div>
        <div class="card-value">{{ validation.metrics_after.validity }}%</div>
        <div class="card-subtext">Before: {{ validation.metrics_before.validity }}%</div>
      </div>
      <div class="card">
        <div class="card-title">Consistency</div>
        <div class="card-value">{{ validation.metrics_after.consistency }}%</div>
        <div class="card-subtext">Before: {{ validation.metrics_before.consistency }}%</div>
      </div>
      <div class="card">
        <div class="card-title">Uniqueness</div>
        <div class="card-value">{{ validation.metrics_after.uniqueness }}%</div>
        <div class="card-subtext">Before: {{ validation.metrics_before.uniqueness }}%</div>
      </div>
    </div>

    <!-- Issue Resolution Summary -->
    <div class="section">
      <h2>Quality Issues & Resolutions Summary</h2>
      <div class="grid" style="grid-template-columns: repeat(4, 1fr); margin-bottom: 0;">
        <div>
          <div class="card-subtext">Detected Issues</div>
          <div style="font-size: 24px; font-weight: 700;">{{ validation.issues_before_count }}</div>
        </div>
        <div>
          <div class="card-subtext">Resolved Issues</div>
          <div style="font-size: 24px; font-weight: 700; color: var(--success);">{{ validation.issues_resolved_count }}</div>
        </div>
        <div>
          <div class="card-subtext">Remaining / Suggestions</div>
          <div style="font-size: 24px; font-weight: 700; color: var(--warning);">{{ validation.issues_remaining_count }}</div>
        </div>
        <div>
          <div class="card-subtext">Newly Introduced Defects</div>
          <div style="font-size: 24px; font-weight: 700; color: var(--success);">{{ validation.issues_newly_introduced_count }}</div>
        </div>
      </div>
    </div>

    <!-- Root Cause Diagnoses -->
    <div class="section">
      <h2>Inferred Root Causes & Diagnostics</h2>
      <table>
        <thead>
          <tr>
            <th>Root Cause Hypothesis</th>
            <th>Affected Columns</th>
            <th>Confidence</th>
            <th>Diagnostic Explanation</th>
            <th>Recommended Action</th>
          </tr>
        </thead>
        <tbody>
          {% for diag in diagnoses %}
          <tr>
            <td><strong>{{ diag.root_cause_label }}</strong></td>
            <td><span class="code">{{ diag.affected_columns | join(', ') }}</span></td>
            <td><span class="badge badge-info">{{ (diag.confidence * 100) | round(0) }}%</span></td>
            <td>{{ diag.explanation }}</td>
            <td style="color: #475569;">{{ diag.recommended_action }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>

    <!-- Change Log Audit Trail -->
    <div class="section">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
        <h2 style="margin: 0; border: none; padding: 0;">Modification Audit Trail</h2>
        <div>
          <span class="badge badge-success">Applied: {{ audit_summary.applied }}</span>
          <span class="badge badge-warning">Suggested: {{ audit_summary.suggested }}</span>
        </div>
      </div>
      <table>
        <thead>
          <tr>
            <th>Change ID</th>
            <th>Row ID</th>
            <th>Column</th>
            <th>Before (Old Value)</th>
            <th>After (New Value)</th>
            <th>Strategy</th>
            <th>Confidence</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {% for entry in audit_entries[:100] %}
          <tr>
            <td><span class="code">{{ entry.change_id }}</span></td>
            <td>{{ entry.row_id }}</td>
            <td><strong>{{ entry.column }}</strong></td>
            <td style="color: var(--danger);"><span class="code">{{ entry.old_value }}</span></td>
            <td style="color: var(--success);"><span class="code">{{ entry.new_value }}</span></td>
            <td>{{ entry.strategy }}</td>
            <td>{{ (entry.confidence * 100) | round(0) }}%</td>
            <td>
              {% if entry.status == 'applied' %}
              <span class="badge badge-success">Applied</span>
              {% elif entry.status == 'suggested' %}
              <span class="badge badge-warning">Suggested</span>
              {% else %}
              <span class="badge badge-info">{{ entry.status }}</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% if audit_entries | length > 100 %}
      <p style="font-size: 12px; color: var(--text-muted); margin-top: 12px;">
        Showing first 100 modifications. Full log available in change_log.csv.
      </p>
      {% endif %}
    </div>

  </div>
</body>
</html>
"""


def generate_html_report(
    validation_result: ValidationResult,
    diagnoses: List[Diagnosis],
    audit_log: AuditLog,
    dataset_name: str = "Dataset",
    output_path: Optional[str] = None
) -> str:
    """Render a comprehensive standalone HTML quality report using Jinja2."""
    from datetime import datetime, timezone

    template = Template(HTML_REPORT_TEMPLATE)
    html_content = template.render(
        dataset_name=dataset_name,
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        validation=validation_result,
        diagnoses=diagnoses,
        audit_summary=audit_log.summary(),
        audit_entries=audit_log.entries
    )

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(html_content)

    return html_content
