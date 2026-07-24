#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/bonus.py
Bonus / Innovation Features (Page 19 of Starter Guide)

Two defensible, non-black-box bonus capabilities:

- Bonus 1: Image Captioning
    (a) A lightweight, offline, TEMPLATE-BASED caption generator that needs
        NO model download and NO internet. It builds a short natural-language
        description from the predicted class + confidence + simple image
        statistics (brightness, contrast, dominant structure).
    (b) An OPTIONAL BLIP neural caption (if `transformers` is installed and a
        model is cached). BLIP is used only as an enrichment; the template
        caption is always available as a graceful fallback.

- Bonus 2: Anomaly / Out-of-Distribution (OOD) Detection
    A principled, model-agnostic uncertainty method based on the softmax
    output alone: Shannon entropy of the predictive distribution + the maximum
    class probability. Inputs whose entropy is high OR whose top probability is
    low are flagged as "possible anomaly / out-of-distribution / rare object".
    No extra training, no external service, fully explainable.

All functions can be integrated into app.py or called independently via the
CLI:  `python src/bonus.py --image X --checkpoint Y`.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ===========================================================================
# Bonus 1a: Offline, template-based captioning (NO model download / internet)
# ===========================================================================

# Per-class descriptive fragments used to compose natural-language captions.
# Each entry: a noun phrase + the salient structural cue humans look for.
CLASS_DESCRIPTORS: Dict[str, Dict[str, str]] = {
    "Spiral Galaxy": {
        "noun": "spiral galaxy",
        "structure": "a bright central core and sweeping arm structure",
    },
    "Elliptical Galaxy": {
        "noun": "elliptical galaxy",
        "structure": "a smooth, featureless oval glow of evenly distributed stars",
    },
    "Nebula": {
        "noun": "nebula",
        "structure": "diffuse clouds of glowing gas and dust",
    },
    "Star Cluster": {
        "noun": "star cluster",
        "structure": "a dense grouping of individually resolved bright stars",
    },
    "Planetary Object": {
        "noun": "planetary object",
        "structure": "a compact, well-defined disk with possible surface or ring detail",
    },
}


def _describe_brightness(mean_luminance: float) -> str:
    """Map a 0-255 mean luminance to a human word. Deterministic."""
    if mean_luminance < 40:
        return "very faint"
    if mean_luminance < 90:
        return "dim"
    if mean_luminance < 160:
        return "moderately bright"
    return "bright"


def _describe_contrast(std_luminance: float) -> str:
    """Map luminance std-dev to a structural-contrast word. Deterministic."""
    if std_luminance < 25:
        return "low-contrast, smooth"
    if std_luminance < 60:
        return "moderately structured"
    return "high-contrast, sharply structured"


def compute_image_stats(image: Union[str, Image.Image, np.ndarray]) -> Dict[str, float]:
    """
    Compute simple, interpretable image statistics used for captioning.

    Args:
        image: file path, PIL image, or HxWxC / HxW numpy array.

    Returns:
        Dict with mean_luminance (0-255), std_luminance, and
        bright_fraction (fraction of pixels brighter than 60% of max).
    """
    if isinstance(image, (str,)):
        arr = np.array(Image.open(image).convert("RGB"), dtype=np.float32)
    elif isinstance(image, Image.Image):
        arr = np.array(image.convert("RGB"), dtype=np.float32)
    else:
        arr = np.asarray(image, dtype=np.float32)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

    # Rec. 601 luma.
    if arr.ndim == 3 and arr.shape[-1] >= 3:
        luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    else:
        luma = arr.reshape(arr.shape[0], -1)

    mean_l = float(luma.mean())
    std_l = float(luma.std())
    bright_fraction = float((luma > 0.6 * 255.0).mean())
    return {
        "mean_luminance": mean_l,
        "std_luminance": std_l,
        "bright_fraction": bright_fraction,
    }


def generate_template_caption(
    class_name: str,
    confidence: float,
    image: Optional[Union[str, Image.Image, np.ndarray]] = None,
) -> Dict[str, str]:
    """
    Build a short natural-language caption WITHOUT any heavy model or internet.

    The caption combines the predicted class, its structural cue, optional
    image statistics (brightness / contrast), and the model confidence, e.g.:

        "A bright, high-contrast, sharply structured spiral galaxy showing a
         bright central core and sweeping arm structure (confidence 0.87)."

    This function is fully deterministic and unit-testable without a checkpoint.

    Args:
        class_name: predicted class display name (e.g. "Spiral Galaxy").
        confidence: top-class probability in [0, 1].
        image: optional path / PIL / ndarray to derive brightness & contrast.

    Returns:
        Dict with 'caption', 'method' ("template"), and 'confidence'.
    """
    desc = CLASS_DESCRIPTORS.get(class_name)
    noun = desc["noun"] if desc else f"astronomical object classified as {class_name}"
    structure = desc["structure"] if desc else "distinctive morphology"

    modifier = ""
    if image is not None:
        try:
            stats = compute_image_stats(image)
            modifier = f"{_describe_brightness(stats['mean_luminance'])}, " \
                       f"{_describe_contrast(stats['std_luminance'])} "
        except Exception as e:  # pragma: no cover - defensive; never fail a caption
            logger.warning(f"Image stats unavailable, using plain template: {e}")

    # Choose "A"/"An" based on the first spoken word (modifier if present, else noun).
    first_word = (modifier if modifier else noun).lstrip()
    article = "An" if first_word[:1].lower() in "aeiou" else "A"
    caption = f"{article} {modifier}{noun} showing {structure} (confidence {confidence:.2f})."
    return {"caption": caption, "method": "template", "confidence": f"{confidence:.2f}"}


# ===========================================================================
# Bonus 1b: Optional BLIP neural captioning (graceful, offline-safe)
# ===========================================================================

def generate_caption(image_path: str, max_length: int = 50) -> Dict[str, str]:
    """
    Generate a neural caption using BLIP, if `transformers` and a local/cached
    model are available. Never raises: returns a dict whose 'model' key is
    "BLIP-base" on success and absent otherwise, so callers can fall back.

    Note: this may download weights on first use; prefer the template caption
    for a guaranteed-offline path.
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError:
        logger.info("transformers not installed; BLIP captioning unavailable.")
        return {"caption": "BLIP unavailable (transformers not installed)", "confidence": "N/A"}

    try:
        import torch  # local import: torch not needed for template path
        from utils import get_device

        device = get_device()  # CUDA > Apple MPS > CPU
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            output_ids = blip_model.generate(
                **inputs, max_length=max_length, num_beams=5, early_stopping=True
            )
        caption = processor.decode(output_ids[0], skip_special_tokens=True)
        logger.info(f"BLIP caption: {caption}")
        return {"caption": caption, "confidence": "generated", "model": "BLIP-base"}
    except Exception as e:
        logger.warning(f"BLIP caption failed, will fall back to template: {e}")
        return {"caption": f"BLIP error: {e}", "confidence": "N/A"}


def generate_caption_with_fallback(
    image_path: str,
    class_name: str,
    confidence: float = 1.0,
    use_blip: bool = False,
) -> Dict[str, str]:
    """
    Return the best available caption.

    Strategy:
      - If `use_blip` and BLIP produces a caption, use it.
      - Otherwise (default) use the offline template caption, which is always
        available and requires no download.

    Args:
        image_path: path to image (used for stats and optional BLIP).
        class_name: predicted class display name.
        confidence: top-class probability, embedded in the caption.
        use_blip: opt-in to the heavier neural captioner.
    """
    if use_blip:
        result = generate_caption(image_path)
        if result.get("model") == "BLIP-base":
            result["method"] = "blip"
            return result
        logger.info("Falling back to offline template caption.")

    return generate_template_caption(class_name, confidence, image=image_path)


# ===========================================================================
# Bonus 2: Anomaly / Out-of-Distribution detection (entropy + max-prob)
# ===========================================================================

# 5-class problem: maximum possible Shannon entropy is log2(5) ~= 2.3219 bits
# (uniform distribution -> maximal uncertainty). A confident, in-distribution
# prediction concentrates mass on one class -> low entropy, high max-prob.
NUM_CLASSES_DEFAULT = 5
MAX_ENTROPY_BITS = math.log2(NUM_CLASSES_DEFAULT)  # ~2.322 for 5 classes

# --- Threshold rationale (documented for defensibility) --------------------
# max_prob_threshold = 0.45:
#   Random guessing over 5 classes gives max-prob 0.20. Well-trained, correct
#   predictions typically exceed 0.60. 0.45 sits comfortably above chance yet
#   below confident territory, so it flags genuinely uncertain inputs (novel
#   morphologies, corrupted frames, off-distribution objects) without spamming
#   confident correct ones.
# entropy_threshold = 1.50 bits (~65% of the 2.322-bit maximum):
#   1.50 bits is roughly the entropy of probability mass spread over ~3 classes.
#   Below it the distribution is peaked (in-distribution); above it the model is
#   effectively hedging across many classes -> likely OOD / rare object.
DEFAULT_MAX_PROB_THRESHOLD = 0.45
DEFAULT_ENTROPY_THRESHOLD = 1.50


def _to_prob_array(probabilities: Union[Dict[str, float], Sequence[float], np.ndarray]) -> np.ndarray:
    """Coerce a dict / list / ndarray of probabilities to a 1-D numpy array."""
    if isinstance(probabilities, dict):
        arr = np.array(list(probabilities.values()), dtype=np.float64)
    else:
        arr = np.asarray(list(probabilities), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("probabilities must be non-empty")
    return arr


def compute_entropy(probabilities: Union[Dict[str, float], Sequence[float], np.ndarray]) -> float:
    """
    Shannon entropy (in bits) of a probability distribution.

    H = -sum(p * log2 p). Zero-probability terms are skipped (0*log0 := 0).
    Higher entropy => more spread-out / uncertain distribution.
    """
    probs = _to_prob_array(probabilities)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def detect_anomaly(
    probabilities: Union[Dict[str, float], Sequence[float], np.ndarray],
    max_prob_threshold: float = DEFAULT_MAX_PROB_THRESHOLD,
    entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
) -> Dict:
    """
    Flag a prediction as a possible anomaly / out-of-distribution / rare object
    using softmax entropy + maximum probability. Purely post-hoc on the model's
    output distribution -- no retraining, no external service, fully explainable.

    A sample is flagged when EITHER:
        max_prob < max_prob_threshold  (model is not confident in any class), OR
        entropy  > entropy_threshold   (probability mass is spread out).

    Args:
        probabilities: dict {class: prob}, or a list/array of probs.
        max_prob_threshold: min top-class prob to be considered in-distribution.
        entropy_threshold: max entropy (bits) to be considered in-distribution.

    Returns:
        {
          "is_anomaly": bool,
          "entropy": float,          # bits
          "max_prob": float,         # top-class probability
          "reason": str,             # human-readable explanation
          "entropy_threshold": float,
          "max_prob_threshold": float,
          "max_entropy_bits": float, # theoretical max for this #classes
        }
    """
    probs = _to_prob_array(probabilities)
    max_prob = float(probs.max())
    entropy = compute_entropy(probs)
    max_entropy = math.log2(len(probs)) if len(probs) > 1 else 0.0

    low_confidence = max_prob < max_prob_threshold
    high_entropy = entropy > entropy_threshold
    is_anomaly = bool(low_confidence or high_entropy)

    reasons: List[str] = []
    if low_confidence:
        reasons.append(f"low confidence (max prob {max_prob:.2f} < {max_prob_threshold:.2f})")
    if high_entropy:
        reasons.append(f"high uncertainty (entropy {entropy:.2f} > {entropy_threshold:.2f} bits)")

    if is_anomaly:
        reason = "Possible anomaly / out-of-distribution / rare object: " + " and ".join(reasons)
    else:
        reason = (f"In-distribution: confident prediction (max prob {max_prob:.2f}, "
                  f"entropy {entropy:.2f} bits)")

    return {
        "is_anomaly": is_anomaly,
        "entropy": entropy,
        "max_prob": max_prob,
        "reason": reason,
        "entropy_threshold": entropy_threshold,
        "max_prob_threshold": max_prob_threshold,
        "max_entropy_bits": max_entropy,
    }


class AnomalyDetector:
    """
    Object wrapper around `detect_anomaly` for callers that hold an
    InferenceResult. Backwards-compatible with earlier revisions.

    Flags images where the softmax distribution indicates the model is
    uncertain (low max-prob or high entropy). See `detect_anomaly` for the
    threshold rationale.
    """

    def __init__(
        self,
        max_prob_threshold: float = DEFAULT_MAX_PROB_THRESHOLD,
        entropy_threshold: float = DEFAULT_ENTROPY_THRESHOLD,
    ):
        self.max_prob_threshold = max_prob_threshold
        self.entropy_threshold = entropy_threshold

    def analyze(self, result) -> Dict:
        """Analyze an InferenceResult (or anything with .all_probabilities)."""
        verdict = detect_anomaly(
            result.all_probabilities,
            max_prob_threshold=self.max_prob_threshold,
            entropy_threshold=self.entropy_threshold,
        )
        verdict["class_name"] = getattr(result, "class_name", None)
        return verdict

    def batch_analyze(self, results: list) -> list:
        return [self.analyze(r) for r in results]


# ===========================================================================
# Bonus 2: Object Localization (Grad-CAM-thresholded pseudo-bounding-box)
# ===========================================================================

def localize_object(
    cam_heatmap: "np.ndarray",
    threshold: float = 0.30,
) -> Dict:
    """
    Derive a pseudo-bounding box from a Grad-CAM heatmap.

    Grad-CAM highlights *where the model looked*. We threshold the normalized
    heatmap and take the tightest bounding box of the resulting binary mask.
    This is NOT a learned detector (no YOLO/Mask R-CNN) -- it is a cheap,
    explainable saliency-localizer that re-uses the classifier's Grad-CAM, which
    the problem statement permits as a weak-form localization under
    "attention or Grad-CAM visualizations" (Bonus Task 4) and is offered here as
    an honest partial answer to Bonus Task 2 (Object Localization).

    Args:
        cam_heatmap: 2-D Grad-CAM heatmap (values in [0, 1] or raw).
        threshold: fraction of the max activation to keep (0.0-1.0).

    Returns:
        {
          "bbox": [x_min, y_min, x_max, y_max] in pixel coords of the heatmap,
          "bbox_frac": same as fractions of width/height (0-1),
          "area_frac": fraction of the image covered by the box,
          "threshold": float,
          "mask_area_px": int,   # how many pixels exceeded threshold,
          "method": "gradcam-threshold",
        }
        If no pixels exceed the threshold, bbox is None and area_frac is 0.
    """
    arr = np.asarray(cam_heatmap, dtype=np.float32)
    if arr.ndim != 2:
        arr = arr.squeeze()
    if arr.ndim != 2:
        raise ValueError(f"cam_heatmap must be 2-D, got shape {arr.shape}")

    mx = float(arr.max())
    if mx <= 0:
        return {"bbox": None, "bbox_frac": None, "area_frac": 0.0,
                "threshold": threshold, "mask_area_px": 0,
                "method": "gradcam-threshold"}

    mask = arr >= (threshold * mx)
    mask_area_px = int(mask.sum())
    h, w = arr.shape
    if mask_area_px == 0:
        return {"bbox": None, "bbox_frac": None, "area_frac": 0.0,
                "threshold": threshold, "mask_area_px": 0,
                "method": "gradcam-threshold"}

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    y_min, y_max = int(np.argmax(rows)), int(len(rows) - 1 - np.argmax(rows[::-1]))
    x_min, x_max = int(np.argmax(cols)), int(len(cols) - 1 - np.argmax(cols[::-1]))

    bbox = [x_min, y_min, x_max, y_max]
    bbox_frac = [x_min / w, y_min / h, x_max / w, y_max / h]
    box_area = max(0, (x_max - x_min)) * max(0, (y_max - y_min))
    area_frac = float(box_area) / float(w * h)
    return {"bbox": bbox, "bbox_frac": bbox_frac, "area_frac": area_frac,
            "threshold": threshold, "mask_area_px": mask_area_px,
            "method": "gradcam-threshold"}


def render_localization_overlay(
    image: "np.ndarray",
    cam_heatmap: "np.ndarray",
    bbox: list,
    color=(0, 255, 0),
    thickness: int = 3,
) -> "np.ndarray":
    """Draw the localization bbox on a copy of `image` (resized to the heatmap)."""
    import cv2
    img = np.array(image)
    if img.dtype != np.uint8:
        img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
    h, w = cam_heatmap.shape[:2]
    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h))
    x1, y1, x2, y2 = bbox
    return cv2.rectangle(img.copy(), (x1, y1), (x2, y2), color, thickness)

def analyze_image_with_bonus(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    generate_caption_flag: bool = True,
    detect_anomalies: bool = True,
    localize: bool = True,
    use_blip: bool = False,
) -> Dict:
    """
    Full analysis pipeline: classify -> caption -> localization -> anomaly/OOD verdict.

    Requires a trained checkpoint (imports inference/gradcam lazily so that the
    caption/anomaly/localization helpers above stay importable without torch).
    """
    from inference import predict_image  # lazy: avoids torch import for helpers

    result = predict_image(image_path, checkpoint_path)
    output: Dict = {"classification": result.to_dict()}

    if generate_caption_flag:
        output["caption"] = generate_caption_with_fallback(
            image_path, result.class_name, confidence=result.confidence, use_blip=use_blip
        )

    if localize:
        try:
            from gradcam import explain_image  # lazy: heavy torch import
            from utils import get_device
            cam_result = explain_image(
                model=_load_model_for_bonus(checkpoint_path),
                image_path=image_path, device=get_device(),
                true_label=result.class_index,
            )
            output["localization"] = localize_object(
                _cam_array_from_overlay(cam_result["predicted_cam"])
            )
        except Exception as e:  # pragma: no cover - defensive; never fail the pipeline
            output["localization"] = {"error": f"localization unavailable: {e}"}

    if detect_anomalies:
        output["anomaly"] = detect_anomaly(result.all_probabilities)

    return output


def _load_model_for_bonus(checkpoint_path: str):
    """Lazy model loader for localization (cached on second call)."""
    from gradcam import load_model_for_gradcam
    from utils import get_device
    if not hasattr(_load_model_for_bonus, "_cache"):
        _load_model_for_bonus._cache = {}
    if checkpoint_path not in _load_model_for_bonus._cache:
        _load_model_for_bonus._cache[checkpoint_path] = load_model_for_gradcam(
            checkpoint_path, get_device()
        )
    return _load_model_for_bonus._cache[checkpoint_path]


def _cam_array_from_overlay(overlay: "np.ndarray") -> "np.ndarray":
    """Best-effort recover a 2-D activation map from a CAM overlay image.

    Overlays are RGB; we take the max over channels as a proxy for activation
    intensity. For exact heatmaps, call `localize_object` directly with the
    raw Grad-CAM array from `gradcam.generate_cam`.
    """
    arr = np.asarray(overlay, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    mx = float(arr.max())
    if mx > 0:
        arr = arr / mx
    return arr


# ===========================================================================
# CLI:  python src/bonus.py --image X --checkpoint Y
# ===========================================================================

def _main() -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="SCALE x ODYSSEY -- Bonus features (caption + localization + OOD)"
    )
    parser.add_argument("--image", required=True, help="Path to an astronomical image")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--use-blip", action="store_true",
                        help="Opt in to neural BLIP caption (may download weights)")
    parser.add_argument("--no-caption", action="store_true", help="Skip caption generation")
    parser.add_argument("--no-anomaly", action="store_true", help="Skip anomaly detection")
    parser.add_argument("--no-localize", action="store_true",
                        help="Skip object localization (Grad-CAM pseudo-bbox)")
    args = parser.parse_args()

    print("=" * 60)
    print("SCALE x ODYSSEY -- Bonus Features")
    print("=" * 60)

    if not os.path.exists(args.image):
        print(f"\n[error] Image not found: {args.image}")
        return 1
    if not os.path.exists(args.checkpoint):
        print(f"\n[error] Checkpoint not found: {args.checkpoint}")
        print("        Train a model first (e.g. `python src/train.py`), or pass")
        print("        --checkpoint <path>. The offline template caption below still")
        print("        works from image statistics alone once a class is known.")
        return 1

    try:
        result = analyze_image_with_bonus(
            args.image,
            args.checkpoint,
            generate_caption_flag=not args.no_caption,
            detect_anomalies=not args.no_anomaly,
            localize=not args.no_localize,
            use_blip=args.use_blip,
        )
    except Exception as e:  # pragma: no cover - CLI safety net
        print(f"\n[error] Analysis failed: {e}")
        return 1

    cls = result["classification"]
    print(f"\nClassification : {cls['class_name']}")
    print(f"Confidence     : {cls['confidence']:.2%}")
    print(f"Inference time : {cls['inference_time_ms']} ms")

    if "caption" in result:
        cap = result["caption"]
        print(f"\nCaption ({cap.get('method', 'N/A')}):")
        print(f"  {cap['caption']}")

    if "localization" in result and isinstance(result["localization"], dict):
        loc = result["localization"]
        if loc.get("bbox") is not None:
            print("\nObject Localization (Grad-CAM pseudo-bbox):")
            print(f"  bbox (px): {loc['bbox']}")
            print(f"  area covered: {loc['area_frac']:.1%} | threshold: {loc['threshold']:.2f} "
                  f"| mask px: {loc['mask_area_px']}")
        elif "error" in loc:
            print(f"\nObject Localization: {loc['error']}")
        else:
            print("\nObject Localization: no salient region above threshold.")

    if "anomaly" in result:
        anom = result["anomaly"]
        verdict = "ANOMALY / OOD" if anom["is_anomaly"] else "in-distribution"
        print(f"\nAnomaly / OOD  : {verdict}")
        print(f"  {anom['reason']}")
        print(f"  entropy = {anom['entropy']:.3f} bits (max {anom['max_entropy_bits']:.3f}), "
              f"max_prob = {anom['max_prob']:.3f}")

    print("\n" + "=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
