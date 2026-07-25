"""Tests for B4 moat features: localization overlay + OOD noise sample."""

import numpy as np

from src.bonus import (
    detect_anomaly,
    localize_object,
    overlay_localization_bbox,
    render_localization_overlay,
)


def test_overlay_localization_bbox_returns_annotated_image():
    """overlay_localization_bbox draws a bbox from a synthetic CAM overlay."""
    image = np.full((50, 50, 3), 128, dtype=np.uint8)
    cam = np.zeros((50, 50), dtype=np.float32)
    cam[10:40, 10:40] = 1.0
    cam_overlay = render_localization_overlay(
        np.full((50, 50, 3), 200, dtype=np.uint8), cam, [10, 10, 39, 39]
    )
    result = overlay_localization_bbox(image, cam_overlay)
    assert "overlay" in result
    assert result["bbox"] is not None
    assert result["area_frac"] > 0.0
    assert result["overlay"].shape == (50, 50, 3)


def test_overlay_localization_empty_cam_returns_none_bbox():
    """When the CAM overlay has no signal, bbox is None, overlay is original."""
    image = np.full((50, 50, 3), 100, dtype=np.uint8)
    cam_overlay = np.zeros((50, 50, 3), dtype=np.uint8)
    result = overlay_localization_bbox(image, cam_overlay)
    assert result["bbox"] is None
    assert result["area_frac"] == 0.0
    assert np.array_equal(result["overlay"], image)


def test_noise_detected_as_anomaly():
    """Synthetic low-confidence probs should trigger the OOD detector."""
    probs = [0.08, 0.22, 0.25, 0.23, 0.22]
    verdict = detect_anomaly(probs)
    assert verdict["is_anomaly"] is True
    assert "low confidence" in verdict["reason"] or "high uncertainty" in verdict["reason"]
