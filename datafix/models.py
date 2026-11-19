"""Data models for DataFix: Issue and ChangeLogEntry."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid


@dataclass
class Issue:
    """Represents a detected data quality issue.
    
    Fields match the exact specification:
    issue_id, issue_type, columns, row_ids, sample_values,
    severity (low/med/high), description, detector, evidence (dict).
    """
    issue_id: str
    issue_type: str
    columns: List[str]
    row_ids: List[int]
    sample_values: List[Any]
    severity: str  # 'low' | 'med' | 'high'
    description: str
    detector: str
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.severity not in {"low", "med", "high"}:
            raise ValueError(f"Invalid severity '{self.severity}'. Must be 'low', 'med', or 'high'.")
        if not self.issue_id:
            self.issue_id = f"ISSUE_{uuid.uuid4().hex[:8]}"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Issue:
        return cls(**data)


@dataclass
class ChangeLogEntry:
    """Represents an atomic proposed or applied modification to a dataset.
    
    Fields match the exact specification:
    change_id, timestamp, row_id, column, old_value, new_value,
    issue_id, strategy, reason (human-readable), confidence,
    status (applied | suggested | approved | rejected | reverted).
    """
    change_id: str
    timestamp: str
    row_id: Any
    column: str
    old_value: Any
    new_value: Any
    issue_id: str
    strategy: str
    reason: str
    confidence: float
    status: str  # 'applied' | 'suggested' | 'approved' | 'rejected' | 'reverted'

    VALID_STATUSES = {"applied", "suggested", "approved", "rejected", "reverted"}

    def __post_init__(self):
        if self.status not in self.VALID_STATUSES:
            raise ValueError(f"Invalid status '{self.status}'. Must be one of {self.VALID_STATUSES}.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(f"Confidence score {self.confidence} must be between 0.0 and 1.0.")
        if not self.change_id:
            self.change_id = f"CHG_{uuid.uuid4().hex[:8]}"
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ChangeLogEntry:
        return cls(**data)
