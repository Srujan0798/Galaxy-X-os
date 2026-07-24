from src.utils import load_config, set_seed, compute_metrics


def test_load_config():
    cfg = load_config()
    assert "data" in cfg
    assert "model" in cfg
    assert cfg["model"]["num_classes"] == 5


def test_set_seed():
    set_seed(42)
    # Should not raise


def test_compute_metrics():
    labels = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    preds = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
    metrics = compute_metrics(labels, preds, num_classes=5)
    assert metrics["accuracy"] == 1.0
    assert "per_class_f1" in metrics
    assert len(metrics["per_class_f1"]) == 5
