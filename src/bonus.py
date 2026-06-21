#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/bonus.py
Bonus Features (Page 19 of Starter Guide)

- Bonus 1: Image Captioning using BLIP (ready-to-call function)
- Bonus 2: Anomaly Detection (low-confidence flagging)

These functions can be integrated into app.py or called independently.
"""

import logging
from typing import Dict, Optional
from pathlib import Path

import torch
import numpy as np
from PIL import Image

from inference import ModelManager, InferenceResult
from utils import get_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Bonus 1: Image Captioning using BLIP
# ---------------------------------------------------------------------------

def generate_caption(image_path: str, max_length: int = 50) -> Dict[str, str]:
    """
    Generate a natural language caption for an astronomical image using BLIP.

    Args:
        image_path: Path to the image file
        max_length: Maximum caption length

    Returns:
        Dict with 'caption' and 'confidence'
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except ImportError:
        logger.error("transformers library not installed. Run: pip install transformers")
        return {"caption": "Captioning unavailable (install transformers library)", "confidence": "N/A"}

    try:
        device = get_device()  # CUDA > Apple MPS > CPU

        # Load BLIP model and processor (auto-downloads on first use)
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)

        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)

        # Generate caption
        with torch.no_grad():
            output_ids = blip_model.generate(
                **inputs,
                max_length=max_length,
                num_beams=5,
                early_stopping=True,
            )

        caption = processor.decode(output_ids[0], skip_special_tokens=True)

        logger.info(f"Generated caption: {caption}")
        return {"caption": caption, "confidence": "generated", "model": "BLIP-base"}

    except Exception as e:
        logger.error(f"Caption generation failed: {e}")
        return {"caption": f"Error: {str(e)}", "confidence": "N/A"}


# Pre-defined captions for each class (fallback when BLIP is unavailable)
CLASS_CAPTIONS = {
    "Spiral Galaxy": "A magnificent spiral galaxy with distinct swirling arms and a bright central bulge.",
    "Elliptical Galaxy": "A smooth elliptical galaxy with an oval shape and uniform stellar distribution.",
    "Nebula": "A colorful nebula with glowing gas clouds and dust illuminated by nearby stars.",
    "Star Cluster": "A dense cluster of bright stars tightly bound together by gravitational forces.",
    "Planetary Object": "A planetary object showing distinct surface features and possible atmospheric details.",
}


def generate_caption_with_fallback(image_path: str, class_name: str) -> Dict[str, str]:
    """
    Generate caption with automatic fallback to template captions if BLIP fails.

    Args:
        image_path: Path to image
        class_name: Predicted class name (for fallback)

    Returns:
        Dict with 'caption', 'confidence', and 'method' (blip or template)
    """
    # Try BLIP first
    result = generate_caption(image_path)

    if result.get("model") == "BLIP-base":
        result["method"] = "blip"
        return result

    # Fallback to template caption
    caption = CLASS_CAPTIONS.get(
        class_name,
        f"An astronomical object classified as {class_name}."
    )
    logger.info(f"Using template caption for {class_name}")
    return {"caption": caption, "confidence": "template", "method": "template"}


# ---------------------------------------------------------------------------
# Bonus 2: Anomaly Detection (Low Confidence Flagging)
# ---------------------------------------------------------------------------

class AnomalyDetector:
    """
    Detects anomalous/unconfident predictions.

    Flags images where:
    - Top prediction confidence < threshold (uncertain classification)
    - Top-2 confidence gap is very small (ambiguous between classes)
    - Entropy of probability distribution is high (high uncertainty)
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        gap_threshold: float = 0.15,
        entropy_threshold: float = 1.0,
    ):
        self.confidence_threshold = confidence_threshold
        self.gap_threshold = gap_threshold
        self.entropy_threshold = entropy_threshold

    def _compute_entropy(self, probabilities: Dict[str, float]) -> float:
        """Compute Shannon entropy of probability distribution."""
        probs = np.array(list(probabilities.values()))
        probs = probs[probs > 0]  # Avoid log(0)
        return float(-np.sum(probs * np.log2(probs)))

    def analyze(self, result: InferenceResult) -> Dict:
        """
        Analyze prediction for anomalies.

        Returns:
            Dict with:
            - is_anomalous: bool
            - confidence_ok: bool
            - gap_ok: bool (top-2 gap is large enough)
            - entropy_ok: bool (entropy is low enough)
            - top_2_gap: float
            - entropy: float
            - recommendation: str
        """
        top_2 = result.top_k[:2]
        gap = top_2[0][1] - top_2[1][1]
        entropy = self._compute_entropy(result.all_probabilities)

        confidence_ok = result.confidence >= self.confidence_threshold
        gap_ok = gap >= self.gap_threshold
        entropy_ok = entropy <= self.entropy_threshold

        is_anomalous = not (confidence_ok and gap_ok and entropy_ok)

        # Build recommendation
        reasons = []
        if not confidence_ok:
            reasons.append(f"low confidence ({result.confidence:.1%} < {self.confidence_threshold:.0%})")
        if not gap_ok:
            reasons.append(f"ambiguous (gap {gap:.1%} < {self.gap_threshold:.0%})")
        if not entropy_ok:
            reasons.append(f"high uncertainty (entropy {entropy:.2f} > {self.entropy_threshold:.1f})")

        if is_anomalous:
            recommendation = f"⚠️ ANOMALY: {result.class_name} — " + ", ".join(reasons)
        else:
            recommendation = f"✅ Normal: {result.class_name} ({result.confidence:.1%} confidence)"

        return {
            "is_anomalous": is_anomalous,
            "confidence_ok": confidence_ok,
            "gap_ok": gap_ok,
            "entropy_ok": entropy_ok,
            "top_2_gap": float(gap),
            "entropy": float(entropy),
            "recommendation": recommendation,
            "details": {
                "confidence": result.confidence,
                "top_prediction": result.class_name,
                "second_prediction": top_2[1][0],
                "second_confidence": top_2[1][1],
            },
        }

    def batch_analyze(self, results: list) -> list:
        """Analyze multiple predictions."""
        return [self.analyze(r) for r in results]


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def analyze_image_with_bonus(
    image_path: str,
    checkpoint_path: str = "checkpoints/best_model.pth",
    generate_caption_flag: bool = True,
    detect_anomalies: bool = True,
) -> Dict:
    """
    Full analysis pipeline: classify + caption + anomaly detection.

    Args:
        image_path: Path to image
        checkpoint_path: Model checkpoint path
        generate_caption_flag: Whether to generate caption
        detect_anomalies: Whether to run anomaly detection

    Returns:
        Complete analysis dict
    """
    from inference import predict_image

    # 1. Classify
    result = predict_image(image_path, checkpoint_path)

    output = {
        "classification": result.to_dict(),
    }

    # 2. Generate caption
    if generate_caption_flag:
        caption_result = generate_caption_with_fallback(image_path, result.class_name)
        output["caption"] = caption_result

    # 3. Anomaly detection
    if detect_anomalies:
        detector = AnomalyDetector()
        anomaly_result = detector.analyze(result)
        output["anomaly"] = anomaly_result

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SCALE x ODYSSEY Bonus Features")
    parser.add_argument("image", type=str, help="Path to image")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--no-caption", action="store_true", help="Skip caption generation")
    parser.add_argument("--no-anomaly", action="store_true", help="Skip anomaly detection")
    args = parser.parse_args()

    print("=" * 60)
    print("SCALE x ODYSSEY -- Bonus Features Demo")
    print("=" * 60)

    result = analyze_image_with_bonus(
        args.image,
        args.checkpoint,
        generate_caption_flag=not args.no_caption,
        detect_anomalies=not args.no_anomaly,
    )

    # Classification
    cls = result["classification"]
    print(f"\n🎯 Classification: {cls['class_name']}")
    print(f"   Confidence: {cls['confidence']:.2%}")
    print(f"   Time: {cls['inference_time_ms']}ms")

    # Caption
    if "caption" in result:
        cap = result["caption"]
        print(f"\n📝 Caption ({cap.get('method', 'N/A')}):")
        print(f"   {cap['caption']}")

    # Anomaly
    if "anomaly" in result:
        anom = result["anomaly"]
        print(f"\n🔍 Anomaly Detection:")
        print(f"   {anom['recommendation']}")
        print(f"   Top-2 gap: {anom['top_2_gap']:.2%}")
        print(f"   Entropy: {anom['entropy']:.3f}")

    print("\n" + "=" * 60)
