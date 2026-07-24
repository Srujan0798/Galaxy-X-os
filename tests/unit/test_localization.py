"""Unit tests for src/bonus.py localization (Bonus Task 2)."""

import numpy as np

from src.bonus import localize_object, render_localization_overlay


def test_localize_centered_blob():
    # A 100x100 heatmap with a 40x40 bright square in the center.
    cam = np.zeros((100, 100), dtype=np.float32)
    cam[30:70, 30:70] = 1.0
    out = localize_object(cam, threshold=0.5)
    assert out["bbox"] == [30, 30, 69, 69]
    assert out["area_frac"] > 0.0
    assert out["mask_area_px"] == 1600
    assert out["method"] == "gradcam-threshold"


def test_localize_empty_heatmap_returns_none_bbox():
    cam = np.zeros((50, 50), dtype=np.float32)
    out = localize_object(cam)
    assert out["bbox"] is None
    assert out["area_frac"] == 0.0
    assert out["mask_area_px"] == 0


def test_localize_threshold_too_high_yields_nothing():
    cam = np.full((40, 40), 0.5, dtype=np.float32)
    # threshold > 1.0 means no pixel can exceed threshold*max (0.5 < 0.5*1.5).
    out = localize_object(cam, threshold=1.5)
    assert out["bbox"] is None


def test_localize_rejects_wrong_shape():
    with __import__("pytest").raises(ValueError):
        localize_object(np.zeros((10, 10, 3)))


def test_render_localization_overlay_draws_bbox():
    cam = np.zeros((50, 50), dtype=np.float32)
    cam[10:40, 10:40] = 1.0
    img = np.full((50, 50, 3), 128, dtype=np.uint8)
    out = localize_object(cam, threshold=0.5)
    rendered = render_localization_overlay(img, cam, out["bbox"], color=(0, 255, 0))
    assert rendered.shape == (50, 50, 3)
    # Green pixels (the bbox edge) should be present.
    green_pixels = np.sum(np.all(rendered == (0, 255, 0), axis=-1))
    assert green_pixels > 0