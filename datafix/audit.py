"""DataFix Audit Trail: Tracks, approves, rejects, and reverts changes with before/after logging."""

from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import pandas as pd

from datafix.models import ChangeLogEntry
from datafix.profiler import safe_assign


class AuditLog:
    """Manages the lifecycle of proposed and applied dataset changes."""

    def __init__(self, entries: Optional[List[ChangeLogEntry]] = None):
        self.entries: List[ChangeLogEntry] = entries or []
        self._entry_map: Dict[str, ChangeLogEntry] = {e.change_id: e for e in self.entries}

    def add_entry(self, entry: ChangeLogEntry) -> None:
        """Add a new ChangeLogEntry to the audit trail."""
        self.entries.append(entry)
        self._entry_map[entry.change_id] = entry

    def get_by_id(self, change_id: str) -> Optional[ChangeLogEntry]:
        """Retrieve an entry by its unique change_id."""
        return self._entry_map.get(change_id)

    def approve(self, change_id: str) -> bool:
        """Approve a suggested change."""
        entry = self.get_by_id(change_id)
        if entry and entry.status in {"suggested", "rejected"}:
            entry.status = "approved"
            return True
        return False

    def reject(self, change_id: str) -> bool:
        """Reject a suggested or approved change."""
        entry = self.get_by_id(change_id)
        if entry and entry.status in {"suggested", "approved"}:
            entry.status = "rejected"
            return True
        return False

    def revert(self, change_id: str, df: pd.DataFrame) -> pd.DataFrame:
        """Revert an applied change on the provided DataFrame without mutating the original."""
        entry = self.get_by_id(change_id)
        if not entry or entry.status != "applied":
            return df.copy()

        df_copy = df.copy()
        row_mask = df_copy["_row_id"] == entry.row_id
        if row_mask.any() and entry.column in df_copy.columns:
            # Restore original value
            safe_assign(df_copy, row_mask, entry.column, entry.old_value)
            entry.status = "reverted"

        return df_copy

    def apply_approved(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all approved suggestions to the DataFrame."""
        df_copy = df.copy()
        for entry in self.entries:
            if entry.status == "approved":
                row_mask = df_copy["_row_id"] == entry.row_id
                if row_mask.any() and entry.column in df_copy.columns:
                    safe_assign(df_copy, row_mask, entry.column, entry.new_value)
                    entry.status = "applied"
        return df_copy

    def filter_by_status(self, status: str) -> List[ChangeLogEntry]:
        """Return entries matching the specified status."""
        return [e for e in self.entries if e.status == status]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the audit entries into a pandas DataFrame."""
        if not self.entries:
            return pd.DataFrame(columns=[
                "change_id", "timestamp", "row_id", "column", "old_value",
                "new_value", "issue_id", "strategy", "reason", "confidence", "status"
            ])
        return pd.DataFrame([e.to_dict() for e in self.entries])

    def to_json(self, indent: int = 2) -> str:
        """Serialize audit log to a JSON string."""
        return json.dumps([e.to_dict() for e in self.entries], indent=indent, default=str)

    def summary(self) -> Dict[str, int]:
        """Return count breakdown by status."""
        counts = {"applied": 0, "suggested": 0, "approved": 0, "rejected": 0, "reverted": 0}
        for e in self.entries:
            counts[e.status] = counts.get(e.status, 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.entries)
