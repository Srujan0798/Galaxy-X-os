"""
Unit tests for src/bonus.py — offline template captioning and
entropy/max-prob anomaly (OOD) detection.

These tests require NO trained checkpoint and NO model download. They exercise
the deterministic template caption and the anomaly logic via mock probability
vectors. Run with `PYTHONPATH=src pytest tests/unit/test_bonus.py`.
"""

import math

import numpy as np
import pytest

from src.bonus import (
    CLASS_DESCRIPTORS,
    MAX_ENTROPY_BITS,
    AnomalyDetector,
    compute_entropy,
    compute_image_stats,
    detect_anomaly,
    generate_template_caption,
)


# --------------------------------------------------------------------------
# Template caption (deterministic, no model)
# --------------------------------------------------------------------------

def test_template_caption_is_deterministic():
    a = generate_template_caption("Spiral Galaxy", 0.87)
    b = generate_template_caption("Spiral Galaxy", 0.87)
    assert a == b
    assert a["method"] == "template"


def test_template_caption_contains_class_confidence_and_structure():
    cap = generate_template_caption("Spiral Galaxy", 0.87)["caption"]
    assert "spiral galaxy" in cap
    assert "0.87" in cap
    assert "arm structure" in cap  # structural cue from CLASS_DESCRIPTORS


def test_template_caption_article_agreement():
    # Vowel-initial noun with no image modifier -> "An".
    cap = generate_template_caption("Elliptical Galaxy", 0.5)["caption"]
    assert cap.startswith("An elliptical galaxy")
    # Consonant-initial noun -> "A".
    cap2 = generate_template_caption("Nebula", 0.5)["caption"]
    assert cap2.startswith("A nebula")


def test_template_caption_unknown_class_graceful():
    cap = generate_template_caption("Quasar", 0.42)
    assert "Quasar" in cap["caption"]
    assert "0.42" in cap["caption"]


def test_template_caption_uses_image_stats():
    # A bright, high-contrast synthetic image should produce brightness/contrast words.
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16] = 255  # half white, half black -> bright regions + high contrast
    cap = generate_template_caption("Nebula", 0.6, image=img)["caption"]
    assert "contrast" in cap or "bright" in cap or "faint" in cap


def test_all_classes_have_descriptors():
    for name in ["Spiral Galaxy", "Elliptical Galaxy", "Nebula",
                 "Star Cluster", "Planetary Object"]:
        assert name in CLASS_DESCRIPTORS


# --------------------------------------------------------------------------
# Image stats
# --------------------------------------------------------------------------

def test_compute_image_stats_black_and_white():
    black = np.zeros((10, 10, 3), dtype=np.uint8)
    white = np.full((10, 10, 3), 255, dtype=np.uint8)
    assert compute_image_stats(black)["mean_luminance"] == pytest.approx(0.0)
    assert compute_image_stats(white)["mean_luminance"] == pytest.approx(255.0, abs=1.0)
    assert compute_image_stats(white)["bright_fraction"] == pytest.approx(1.0)


def test_compute_image_stats_grayscale_2d():
    gray = np.full((8, 8), 128, dtype=np.uint8)
    stats = compute_image_stats(gray)
    assert stats["mean_luminance"] == pytest.approx(128.0, abs=1.0)


# --------------------------------------------------------------------------
# Entropy
# --------------------------------------------------------------------------

def test_entropy_uniform_is_maximal():
    uniform = [0.2, 0.2, 0.2, 0.2, 0.2]
    assert compute_entropy(uniform) == pytest.approx(MAX_ENTROPY_BITS, abs=1e-9)
    assert MAX_ENTROPY_BITS == pytest.approx(math.log2(5))


def test_entropy_one_hot_is_zero():
    assert compute_entropy([1.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(0.0)


def test_entropy_accepts_dict():
    d = {"a": 0.5, "b": 0.5}
    assert compute_entropy(d) == pytest.approx(1.0)


def test_entropy_empty_raises():
    with pytest.raises(ValueError):
        compute_entropy([])


# --------------------------------------------------------------------------
# Anomaly / OOD detection
# --------------------------------------------------------------------------

def test_confident_prediction_is_not_anomaly():
    v = detect_anomaly([0.90, 0.04, 0.03, 0.02, 0.01])
    assert v["is_anomaly"] is False
    assert v["max_prob"] == pytest.approx(0.90)
    assert "In-distribution" in v["reason"]


def test_uniform_prediction_is_anomaly():
    v = detect_anomaly([0.2, 0.2, 0.2, 0.2, 0.2])
    assert v["is_anomaly"] is True
    # Both triggers fire for a uniform distribution.
    assert "low confidence" in v["reason"]
    assert "high uncertainty" in v["reason"]


def test_low_confidence_triggers_anomaly():
    # Max prob 0.40 < 0.45 threshold, but entropy still under limit.
    v = detect_anomaly([0.40, 0.40, 0.20, 0.0, 0.0])
    assert v["is_anomaly"] is True
    assert v["max_prob"] == pytest.approx(0.40)


def test_result_has_required_structured_keys():
    v = detect_anomaly([0.6, 0.1, 0.1, 0.1, 0.1])
    for key in ("is_anomaly", "entropy", "max_prob", "reason"):
        assert key in v
    assert isinstance(v["is_anomaly"], bool)
    assert isinstance(v["entropy"], float)
    assert isinstance(v["max_prob"], float)
    assert isinstance(v["reason"], str)


def test_custom_thresholds_respected():
    # With a very lax max-prob threshold, a 0.5 top prob is in-distribution.
    v = detect_anomaly([0.5, 0.3, 0.2, 0.0, 0.0],
                       max_prob_threshold=0.3, entropy_threshold=2.5)
    assert v["is_anomaly"] is False


def test_detect_anomaly_accepts_dict():
    d = {"Spiral Galaxy": 0.85, "Nebula": 0.05, "Star Cluster": 0.05,
         "Elliptical Galaxy": 0.03, "Planetary Object": 0.02}
    v = detect_anomaly(d)
    assert v["is_anomaly"] is False
    assert v["max_prob"] == pytest.approx(0.85)


# --------------------------------------------------------------------------
# AnomalyDetector wrapper (works with a mock InferenceResult)
# --------------------------------------------------------------------------

class _MockResult:
    def __init__(self, probs, class_name="Spiral Galaxy"):
        self.all_probabilities = probs
        self.class_name = class_name


def test_anomaly_detector_wrapper():
    det = AnomalyDetector()
    confident = _MockResult({"a": 0.9, "b": 0.04, "c": 0.03, "d": 0.02, "e": 0.01})
    uncertain = _MockResult({"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2})
    assert det.analyze(confident)["is_anomaly"] is False
    assert det.analyze(uncertain)["is_anomaly"] is True
    assert det.analyze(confident)["class_name"] == "Spiral Galaxy"


def test_anomaly_detector_batch():
    det = AnomalyDetector()
    results = [
        _MockResult({"a": 0.9, "b": 0.1, "c": 0.0, "d": 0.0, "e": 0.0}),
        _MockResult({"a": 0.2, "b": 0.2, "c": 0.2, "d": 0.2, "e": 0.2}),
    ]
    verdicts = det.batch_analyze(results)
    assert [v["is_anomaly"] for v in verdicts] == [False, True]
