"""Unit tests for src/prepare_data.py pure helpers (no network / no torch)."""

import numpy as np

from src.prepare_data import dedupe, md5_array, to_rgb_uint8, compute_class_weights


def test_to_rgb_uint8_resizes_and_returns_uint8():
    arr = np.zeros((100, 100, 3), dtype=np.uint8)
    out = to_rgb_uint8(arr, 224)
    assert out.shape == (224, 224, 3)
    assert out.dtype == np.uint8


def test_to_rgb_uint8_coerces_float():
    arr = np.zeros((50, 50), dtype=np.float32)
    out = to_rgb_uint8(arr, 64)
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_dedupe_removes_exact_duplicates():
    a = np.full((32, 32, 3), 10, dtype=np.uint8)
    b = np.full((32, 32, 3), 20, dtype=np.uint8)
    out = dedupe([a, a, b, b, b])
    assert len(out) == 2


def test_md5_array_is_deterministic():
    a = np.zeros((16, 16, 3), dtype=np.uint8)
    assert md5_array(a) == md5_array(a.copy())


def test_compute_class_weights_balanced_is_uniform():
    # 5 classes, 8 train images each -> all equal weight.
    split_data = {"train": [(c, None) for c in
                            ["spiral_galaxy", "elliptical_galaxy", "nebula",
                             "star_cluster", "planetary"] for _ in range(8)]}
    w = compute_class_weights(split_data)
    vals = list(w.values())
    assert max(vals) - min(vals) < 1e-6


def test_compute_class_weights_imbalanced_gives_minority_higher_weight():
    split_data = {
        "train": [("spiral_galaxy", None)] * 10
                 + [("nebula", None)] * 100
                 + [("elliptical_galaxy", None)] * 100
                 + [("star_cluster", None)] * 100
                 + [("planetary", None)] * 100,
    }
    w = compute_class_weights(split_data)
    assert w["spiral_galaxy"] >= w["nebula"]