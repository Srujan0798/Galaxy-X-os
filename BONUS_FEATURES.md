# Bonus Features — Galaxy-X-os (SCALE x ODYSSEY)

This document describes the **optional bonus tasks** implemented in this project, the
approach each one takes, exactly how to run it, and its honest limitations.

All bonus logic lives in [`src/bonus.py`](src/bonus.py), built on top of the inference
engine in [`src/inference.py`](src/inference.py) and the Grad-CAM module in
[`src/gradcam.py`](src/gradcam.py).

> Device note: all bonus code selects the accelerator via `get_device()` in
> `src/utils.py`, which prefers **CUDA > Apple MPS > CPU**.

---

## Summary: what is and isn't implemented

| Bonus task | Status | Approach |
|------------|--------|----------|
| 1. Image Captioning | **Implemented** | Offline template captioner (`generate_template_caption`) with optional BLIP (`Salesforce/blip-image-captioning-base`) |
| 2. Object Localization | **Implemented (weak-form)** | `localize_object` derives a bounding box by thresholding the classifier's Grad-CAM heatmap. NOT a learned detector (no YOLO / Mask R-CNN). |
| 3. Anomaly Detection | **Implemented (heuristic)** | `detect_anomaly` flags low max-prob OR high Shannon entropy over the softmax output |
| 4. Interactive Web App | **Implemented** | Streamlit app under `app/` |

---

## Bonus 1 — Image Captioning

**Where:** `src/bonus.py`
- `generate_template_caption(class_name, confidence, image=None)` — the default, offline, deterministic path
- `generate_caption(image_path)` — optional BLIP neural caption
- `generate_caption_with_fallback(image_path, class_name, confidence, use_blip=False)`
- `CLASS_DESCRIPTORS` — per-class structural cues

**Approach.**
The default path composes a short natural-language caption from the predicted
class + its structural cue + image brightness/contrast words + confidence,
e.g.: *"A bright, high-contrast, sharply structured spiral galaxy showing a
bright central core and sweeping arm structure (confidence 0.87)."* It is fully
deterministic and unit-tested without a checkpoint.

If `use_blip=True` and `transformers` is installed, `Salesforce/blip-image-captioning-base`
generates a neural caption via beam search (`num_beams=5`, `early_stopping=True`).
On any failure (missing transformers, network error, generation error) it
falls back to the template path. The returned dict's `method` field is
`"template"` or `"blip"`.

**How to run.**
```bash
# Full bonus pipeline (classify + caption + localization + anomaly):
python src/bonus.py path/to/image.jpg

# Caption only:
python src/bonus.py path/to/image.jpg --no-anomaly --no-localize

# Opt into BLIP:
python src/bonus.py path/to/image.jpg --use-blip
```

**Honest limitations.**
- The template caption is a composed sentence, not a domain-tuned neural
  description. It conveys class + structural cue + brightness/contrast + confidence.
- BLIP is a generic natural-image captioner with **no astronomical training**;
  captions for telescope imagery are often vague ("a black and white photo of a
  star"). Demonstrated as a capability, not a domain-tuned result.
- First BLIP use downloads ~1 GB of weights (needs network).
- The model is reloaded on every `generate_caption()` call (no caching), so batch
  captioning via BLIP is slow.

---

## Bonus 2 — Object Localization (Grad-CAM pseudo-bounding-box)

**Where:** `src/bonus.py`
- `localize_object(cam_heatmap, threshold=0.30) -> dict`
- `render_localization_overlay(image, cam_heatmap, bbox, ...) -> np.ndarray`

**Approach.**
This is a **saliency-threshold localizer**, NOT a learned detector (no YOLO,
Mask R-CNN, or U-Net). It re-uses the classifier's Grad-CAM — the same mechanism
the problem statement permits under *"attention or Grad-CAM visualizations"*
(Bonus Task 4) — and offers it as an honest partial answer to Bonus Task 2
(Object Localization).

`localize_object` normalizes the 2-D Grad-CAM heatmap to [0, 1], keeps pixels
≥ `threshold * max`, and returns the tightest bounding box of the resulting
binary mask as `[x_min, y_min, x_max, y_max]` in heatmap-pixel coordinates, plus
`bbox_frac` (0–1), `area_frac`, and `mask_area_px`.

**How to run.**
```python
from src.bonus import localize_object, render_localization_overlay
from src.gradcam import generate_cam
import numpy as np

cam = generate_cam(model, input_tensor, target_class=0, device=device)  # 2-D heatmap
loc = localize_object(cam, threshold=0.30)
print(loc["bbox"], loc["area_frac"])
overlay = render_localization_overlay(image_rgb, cam, loc["bbox"])
```

Or via the CLI (which integrates localization into the bonus pipeline):
```bash
python src/bonus.py path/to/image.jpg   # prints bbox + area
```

**Honest limitations.**
- This is a weak-form localizer: it locates the **most salient region**, not
  individually detected objects. Multiple objects in one frame collapse to one box.
- A real detector (YOLO / Mask R-CNN) would be the strong-form answer to Bonus
  Task 2; that is out of scope for this submission but the Grad-CAM foundation is
  already here and a detector could be bolted on without disturbing the pipeline.
- Box tightness depends on `threshold`; default 0.30 works well on the committed
  Grad-CAM samples but may need tuning on noisier inputs.

---

## Bonus 3 — Anomaly / Out-of-Distribution Detection

**Where:** `src/bonus.py`
- `detect_anomaly(probabilities, max_prob_threshold=0.45, entropy_threshold=1.50) -> dict`
- `AnomalyDetector` class wrapper for batch use
- `compute_entropy(probabilities) -> float`

**Approach.**
A sample is flagged as a possible anomaly / OOD / rare object when **either**:
- `max_prob < 0.45` (model is not confident in any single class), **or**
- `entropy > 1.50 bits` (probability mass is spread over ~3+ classes).

5-class maximum entropy is `log2(5) ≈ 2.322` bits. The thresholds are documented
in `bonus.py` and chosen to flag genuinely uncertain inputs without spamming
confident correct ones. Purely post-hoc on the softmax — no retraining, no
external service, fully explainable.

**How to run.**
```python
from src.bonus import detect_anomaly, AnomalyDetector
from src.inference import predict_image

result = predict_image("image.jpg")
verdict = detect_anomaly(result.all_probabilities)
print(verdict["is_anomaly"], verdict["reason"])

det = AnomalyDetector(max_prob_threshold=0.45, entropy_threshold=1.50)
print(det.analyze(result)["is_anomaly"])
```

**Honest limitations.**
- No dedicated OOD evaluation set is committed; thresholds are principled but
  not tuned on held-out non-astro images.
- This is a softmax-uncertainty proxy, not a density-estimation OOD method
  (no Mahalanobis, no energy score).

---

## Bonus 4 — Interactive Web App

**Where:** `app/app.py` (Streamlit)

The web app integrates classification + Grad-CAM + template caption + anomaly
verdict. Run with `streamlit run app/app.py`. See `README.md` for details.