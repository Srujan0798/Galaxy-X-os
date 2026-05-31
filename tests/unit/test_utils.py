import pytest
from src.utils import load_config, set_seed, compute_metrics


def test_load_config():
    cfg = load_config("config/config.yaml")
    assert "data" in cfg
    assert "model" in cfg


def test_set_seed():
    set_seed(42)
    # Should not raise


def test_compute_metrics():
    metrics = compute_metrics([0,1,2], [0,1,2], num_classes=5)
    assert metrics["accuracy"] == 1.0
