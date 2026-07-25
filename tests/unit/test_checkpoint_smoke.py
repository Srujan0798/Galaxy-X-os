"""Smoke test: load checkpoint, run inference on one sample, verify outputs.

Skips gracefully when checkpoints/best_model.pth is absent.
"""

import os
from pathlib import Path

import pytest

from src.dataset import CLASS_NAMES_DISPLAY
from src.inference import ModelManager

CKPT = "checkpoints/best_model.pth"
SAMPLES = Path("data/samples")


def _first_sample() -> str:
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        matches = sorted(SAMPLES.rglob(ext))
        if matches:
            return str(matches[0])
    return ""


def _ckpt_exists() -> bool:
    if not os.path.isfile(CKPT):
        return False
    sample = _first_sample()
    return bool(sample) and os.path.isfile(sample)


@pytest.mark.skipif(not _ckpt_exists(), reason="No checkpoint or samples found")
def test_checkpoint_predict_smoke():
    sample = _first_sample()
    manager = ModelManager(CKPT)
    result = manager.predict(sample)

    assert result.class_name in CLASS_NAMES_DISPLAY
    assert 0 < result.confidence <= 1.0
    assert result.inference_time_ms < 15000

    # Also verify top-k structure
    assert len(result.top_k) == len(CLASS_NAMES_DISPLAY)
    assert len(result.top_k) >= 3
