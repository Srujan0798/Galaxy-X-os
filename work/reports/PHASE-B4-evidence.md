# B4 — Competitive Moat Features — Evidence

## Features Added

### 1. Localization Overlay on Grad-CAM

**File:** `app/app.py` — `render_localization()` (new function)

After Grad-CAM is rendered, an optional checkbox `"📍 Show localization bounding box"` appears. When checked, it calls `bonus.overlay_localization_bbox()` which:

1. Extracts a 2-D activation proxy from the CAM overlay via `_cam_array_from_overlay()`
2. Runs `localize_object()` to threshold the heatmap at 0.30
3. Draws the tight bounding box on the original image via `render_localization_overlay()`

**Key source:** `src/bonus.py:528` — new `overlay_localization_bbox()` convenience function.

**Test:** `tests/unit/test_b4_moat.py::test_overlay_localization_bbox_returns_annotated_image`
**Test:** `tests/unit/test_b4_moat.py::test_overlay_localization_empty_cam_returns_none_bbox`

### 2. OOD "Reject" Story — Synthetic Noise Sample

**File:** `data/samples/noise/noise_1.png` — 224×224 RGB uniform noise (seed 42).

**File:** `app/app.py:378-403` — in the `elif samples:` block, a dedicated `"🧪 OOD / Anomaly Detection Demo"` section with a `"🌀 Try noise sample"` button.

When clicked, the full pipeline runs: prediction → Grad-CAM → caption → **anomaly detection**. The random noise produces an approximately uniform softmax distribution, which triggers the anomaly detector (max-prob < 0.45 and entropy > 1.50 bits), showing the ⚠️ warning.

**Test:** `tests/unit/test_b4_moat.py::test_noise_image_exists_and_is_loadable`
**Test:** `tests/unit/test_b4_moat.py::test_noise_detected_as_anomaly`

## Files Changed

| File | Change |
|---|---|
| `src/bonus.py` | Added `overlay_localization_bbox()` (line 528) — end-to-end bbox overlay from CAM overlay |
| `app/app.py` | `render_gradcam` returns result dict; new `render_localization()` function; OOD noise button section; both call sites updated |
| `data/samples/noise/noise_1.png` | New synthetic noise image (created) |
| `tests/unit/test_b4_moat.py` | 4 new tests covering both moat features |

## Golden Path Check

- Existing tests (`test_bonus.py` + `test_localization.py` = 25 tests) still pass ✓
- No existing modules restructured
- Phase 7 UI untouched (only additions)
