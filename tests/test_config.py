"""Tests for DataFix configuration."""

import pytest
from datafix.config import Config


def test_default_config():
    cfg = Config()
    assert cfg.seed == 42
    assert cfg.auto_apply_threshold == 0.85
    assert cfg.similarity_threshold == 0.85
    assert "n/a" in cfg.disguised_nulls
    assert pytest.approx(
        cfg.weight_completeness + cfg.weight_validity + cfg.weight_uniqueness + cfg.weight_consistency
    ) == 1.0


def test_config_serialization():
    cfg = Config(seed=123, auto_apply_threshold=0.90)
    d = cfg.to_dict()
    assert d["seed"] == 123
    assert d["auto_apply_threshold"] == 0.90
    
    restored = Config.from_dict(d)
    assert restored.seed == 123
    assert restored.auto_apply_threshold == 0.90


def test_invalid_config_weights():
    with pytest.raises(ValueError, match="Quality weights must sum to 1.0"):
        Config(weight_completeness=0.5, weight_validity=0.5, weight_uniqueness=0.5, weight_consistency=0.5)
