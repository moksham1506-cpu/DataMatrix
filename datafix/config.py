"""Configuration settings for DataFix."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Set


DEFAULT_DISGUISED_NULLS: List[str] = [
    "n/a", "na", "null", "none", "-", "--", "---", "?", "??", "???",
    "unknown", "missing", "nan", "nil", "n.a.", "#n/a", "#na",
    "void", "empty", "not available", "not applicable", "undefined"
]


@dataclass
class Config:
    """Global configuration for DataFix detection, diagnosis, repair, and validation."""
    seed: int = 42
    auto_apply_threshold: float = 0.85
    similarity_threshold: float = 0.85
    category_similarity_threshold: float = 0.82
    outlier_iqr_multiplier: float = 1.5
    outlier_zscore_threshold: float = 3.0
    outlier_contamination: float = 0.03
    enable_llm_explanation: bool = False
    
    # Data Quality Score weights (must sum to 1.0)
    weight_completeness: float = 0.30
    weight_validity: float = 0.30
    weight_uniqueness: float = 0.20
    weight_consistency: float = 0.20
    
    # Disguised null tokens (normalized lowercase)
    disguised_nulls: List[str] = field(default_factory=lambda: list(DEFAULT_DISGUISED_NULLS))

    def __post_init__(self):
        total_weight = (
            self.weight_completeness
            + self.weight_validity
            + self.weight_uniqueness
            + self.weight_consistency
        )
        if not (0.99 <= total_weight <= 1.01):
            raise ValueError(f"Quality weights must sum to 1.0, got {total_weight:.4f}")
        if not (0.0 <= self.auto_apply_threshold <= 1.0):
            raise ValueError("auto_apply_threshold must be between 0.0 and 1.0")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Config:
        return cls(**data)
