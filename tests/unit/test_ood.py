"""Cross-survey / OOD evaluation of the anomaly detector.

We don't have a trained model checkpoint in CI, but the anomaly detector is a
pure function of the softmax distribution. We test it on:

1. IN-distribution-like confident predictions (should be flagged in-dist).
2. Deliberately uniform / hedged distributions (should be flagged OOD).
3. Edge-case distributions near the threshold boundaries.

This is a behavioural OOD test of the *detector*, not an end-to-end model OOD
evaluation (which would require the checkpoint + held-out Hubble imagery).
"""

import numpy as np

from src.bonus import detect_anomaly, compute_entropy, MAX_ENTROPY_BITS


def test_in_distribution_confident_not_flagged():
    # A confident, correct-looking 5-class prediction.
    probs = np.array([0.02, 0.90, 0.04, 0.02, 0.02])
    v = detect_anomaly(probs)
    assert v["is_anomaly"] is False
    assert v["max_prob"] > 0.45
    assert v["entropy"] < 1.50


def test_ood_uniform_distribution_flagged():
    probs = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
    v = detect_anomaly(probs)
    assert v["is_anomaly"] is True
    assert abs(compute_entropy(probs) - MAX_ENTROPY_BITS) < 1e-9


def test_ood_hedged_between_two_classes_flagged():
    # 0.50 / 0.50 split -> entropy = 1.0 bit (under 1.50) and max_prob = 0.50
    # (above 0.45) -> NOT flagged. A 3-way spread is needed to exceed 1.50 bits.
    probs = np.array([0.50, 0.50, 0.00, 0.00, 0.00])
    v = detect_anomaly(probs)
    assert v["is_anomaly"] is False  # confident two-way split, not OOD

    # A genuine 3-way hedge (0.34/0.33/0.33) -> entropy ~ 1.58 > 1.50 -> flagged.
    probs3 = np.array([0.34, 0.33, 0.33, 0.00, 0.00])
    v3 = detect_anomaly(probs3)
    assert v3["is_anomaly"] is True


def test_ood_low_confidence_single_peak_flagged():
    # Peak at 0.40 (below 0.45) but rest spread.
    probs = np.array([0.40, 0.30, 0.20, 0.05, 0.05])
    v = detect_anomaly(probs)
    assert v["is_anomaly"] is True  # low confidence trigger


def test_boundary_just_above_threshold_not_flagged():
    # max_prob 0.85 and a very peaked distribution -> entropy well under 1.50.
    probs = np.array([0.85, 0.10, 0.03, 0.01, 0.01])
    v = detect_anomaly(probs)
    assert v["is_anomaly"] is False  # confident and peaked -> in-dist


def test_random_seed_reproducible_verdicts():
    rng = np.random.default_rng(42)
    for _ in range(20):
        p = rng.dirichlet(np.ones(5))
        v1 = detect_anomaly(p)
        v2 = detect_anomaly(p)
        assert v1["is_anomaly"] == v2["is_anomaly"]
        assert v1["reason"] == v2["reason"]