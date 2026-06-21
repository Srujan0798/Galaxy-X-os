# Bonus Features — Galaxy-X-os (SCALE x ODYSSEY)

This document describes the **optional bonus tasks** implemented in this project, the
approach each one takes, exactly how to run it, and its honest limitations.

All bonus logic lives in [`src/bonus.py`](src/bonus.py), built on top of the inference
engine in [`src/inference.py`](src/inference.py).

> Device note: all bonus code selects the accelerator via `get_device()` in
> `src/utils.py`, which prefers **CUDA > Apple MPS > CPU**. On the target machine
> (8 GB MacBook Air, Apple Silicon, no CUDA) this correctly uses the MPS GPU.

---

## Summary: what is and isn't implemented

| Bonus task | Status | Approach |
|------------|--------|----------|
| 1. Image Captioning | **Implemented** | Pre-trained BLIP (`Salesforce/blip-image-captioning-base`) via `transformers`, with a per-class template fallback |
| 2. Object Localization (bounding boxes / segmentation) | **Not implemented as boxes/masks** | No detector/segmenter. The closest existing capability is Grad-CAM saliency in `src/gradcam.py` (separate module, not part of `bonus.py`) |
| 3. Anomaly Detection | **Implemented (heuristic)** | Confidence + top-2 gap + Shannon entropy thresholds over the classifier's softmax output |
| 4. Interactive Web App | Out of scope here | Handled by the Streamlit app under `app/` (not documented in this file) |

---

## Bonus 1 — Image Captioning

**Where:** `src/bonus.py`
- `generate_caption(image_path, max_length=50)`
- `generate_caption_with_fallback(image_path, class_name)`
- `CLASS_CAPTIONS` (template dictionary)

**Approach.**
Captioning uses the off-the-shelf **BLIP** model
`Salesforce/blip-image-captioning-base` from Hugging Face `transformers`. The image
is opened with PIL, processed by `BlipProcessor`, and a caption is generated with
beam search (`num_beams=5`, `early_stopping=True`). The model is moved to the device
chosen by `get_device()`.

If `transformers` is not installed **or** BLIP generation raises any error,
`generate_caption_with_fallback()` falls back to a **fixed template caption** keyed on
the predicted class (e.g. *"A magnificent spiral galaxy with distinct swirling
arms..."*). The returned dict records which path was used via the `method` field
(`"blip"` or `"template"`).

**How to run.**
```bash
# As part of the full bonus pipeline (classify + caption + anomaly):
python src/bonus.py path/to/image.jpg

# Caption only (skip anomaly detection):
python src/bonus.py path/to/image.jpg --no-anomaly

# Programmatic use:
python -c "from src.bonus import generate_caption_with_fallback; \
print(generate_caption_with_fallback('path/to/image.jpg', 'Spiral Galaxy'))"
```

**Honest limitations.**
- BLIP is a **generic natural-image** captioner; it has **no astronomical training**.
  Captions for telescope imagery are often vague or off-domain ("a black and white
  photo of a star"). This is a demonstration of the capability, not a domain-tuned
  result.
- First run **downloads ~1 GB** of model weights from Hugging Face (needs network);
  loading BLIP on an 8 GB machine adds noticeable memory pressure and latency.
- The template fallback is **not a real caption** — it is a canned sentence per class
  and conveys no information about the specific image.
- The model is reloaded on every `generate_caption()` call (no caching), so batch
  captioning is slow.

---

## Bonus 2 — Object Localization (NOT implemented as detection/segmentation)

There is **no bounding-box detector or segmentation model** in `src/bonus.py` or
`src/inference.py`. To be honest about scope: this bonus task is **not implemented**.

The nearest related capability is **Grad-CAM saliency**, implemented separately in
`src/gradcam.py` (using the `grad-cam` package). It produces a heatmap overlay showing
which image regions most influenced the classifier's decision. This is *explainability*,
not localization — it yields no boxes or pixel masks — but it can visually indicate
*where* the object of interest sits.

**How to run (Grad-CAM, the adjacent capability):**
```bash
make gradcam        # or: python src/gradcam.py
```

If true localization is required, it would need a new detection/segmentation head
(e.g. YOLO/Detectron-style boxes or a U-Net mask) — none exists today.

---

## Bonus 3 — Anomaly Detection

**Where:** `src/bonus.py` → `class AnomalyDetector`
- `analyze(result: InferenceResult) -> Dict`
- `batch_analyze(results: list) -> list`

**Approach (heuristic, no separate anomaly model).**
This is **not** an out-of-distribution detector trained on data. It is a lightweight
**heuristic over the classifier's softmax output** that flags predictions the model is
unsure about. An image is flagged anomalous if **any** of these fail:

1. **Confidence** — top-class probability `< confidence_threshold` (default `0.5`).
2. **Top-2 gap** — gap between the top two classes `< gap_threshold` (default `0.15`),
   i.e. the model is torn between two classes.
3. **Entropy** — Shannon entropy of the 5-class distribution `> entropy_threshold`
   (default `1.0`), i.e. probability mass is spread out.

The result dict includes `is_anomalous`, the individual `*_ok` flags, `top_2_gap`,
`entropy`, a human-readable `recommendation`, and supporting `details`.

**How to run.**
```bash
# Full pipeline (classify + caption + anomaly):
python src/bonus.py path/to/image.jpg

# Anomaly only (skip captioning):
python src/bonus.py path/to/image.jpg --no-caption

# Programmatic use:
python -c "from src.inference import predict_image; from src.bonus import AnomalyDetector; \
r = predict_image('path/to/image.jpg'); print(AnomalyDetector().analyze(r))"
```

**Honest limitations.**
- This detects **low-confidence / ambiguous in-distribution** predictions, not genuine
  novelty. A truly out-of-distribution image (e.g. a cat photo) that the model
  confidently misclassifies as a galaxy will **not** be flagged — softmax confidence is
  unreliable on OOD inputs.
- The thresholds are **hand-picked defaults**, not calibrated against a validation set.
  They will need tuning per checkpoint to be meaningful.
- Entropy is computed in **bits** (log base 2) over 5 classes; max possible entropy is
  `log2(5) ≈ 2.32`, so the default `1.0` threshold is a reasonable but unvalidated
  midpoint.

---

## Full pipeline entry point

`analyze_image_with_bonus(image_path, checkpoint_path="checkpoints/best_model.pth", ...)`
in `src/bonus.py` runs classification, then (optionally) captioning and anomaly
detection, returning a single combined dict. The CLI (`python src/bonus.py <image>`)
wraps this and pretty-prints the results.

```bash
python src/bonus.py path/to/image.jpg \
    --checkpoint checkpoints/best_model.pth \
    [--no-caption] [--no-anomaly]
```

**Requirements:**
- A trained checkpoint at `checkpoints/best_model.pth` (present in this repo).
- `transformers>=4.30.0` (declared in `requirements.txt`) for BLIP captioning;
  without it, captioning degrades to the template fallback.
