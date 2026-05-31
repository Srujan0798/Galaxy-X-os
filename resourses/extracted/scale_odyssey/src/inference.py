#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- Fast Inference Module

Provides single-image and batch inference with timing.
Loads best_model.pth and returns predictions ready for display.

Designed for:
- Script-based batch processing
- Web demo backend (Streamlit/Gradio/FastAPI)
- Real-time classification with Grad-CAM

Target: <5 seconds per image on consumer GPU/CPU.
"""

import os
import time
import logging
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import AstroClassifier
from augmentations import CLASS_NAMES_DISPLAY, get_validation_augmentations

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result Data Class
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    """Structured result from a single inference call."""
    class_name: str
    class_index: int
    confidence: float
    all_probabilities: Dict[str, float]
    inference_time_ms: float
    top_k: List[Tuple[str, float]]  # sorted (class_name, prob) pairs

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "class_name": self.class_name,
            "class_index": self.class_index,
            "confidence": self.confidence,
            "all_probabilities": self.all_probabilities,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "top_3": [(name, round(float(prob), 4)) for name, prob in self.top_k[:3]],
        }


# ---------------------------------------------------------------------------
# Model Manager (cache-friendly)
# ---------------------------------------------------------------------------

class ModelManager:
    """
    Manages model loading with optional singleton caching.

    Usage:
        # Cached (for web apps)
        manager = ModelManager.singleton()

        # Fresh instance
        manager = ModelManager(checkpoint_path="checkpoints/best_model.pth")
    """

    _instance: Optional["ModelManager"] = None

    def __init__(
        self,
        checkpoint_path: str = "checkpoints/best_model.pth",
        device: Optional[str] = None,
        compile_model: bool = False,
    ):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[nn.Module] = None
        self.image_size: int = 224
        self._transform = get_validation_augmentations(self.image_size)
        self._warmup_done: bool = False

        self._load_model()

        if compile_model and hasattr(torch, "compile"):
            logger.info("Compiling model with torch.compile()...")
            self.model = torch.compile(self.model)

    def _load_model(self):
        """Load model from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found: {self.checkpoint_path}\n"
                f"Train first: python src/train.py"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        config = checkpoint.get("config", {})
        model_cfg = config.get("model", {})
        backbone = model_cfg.get("backbone", "efficientnet_b3")
        num_classes = model_cfg.get("num_classes", 5)
        dropout = model_cfg.get("dropout", 0.4)
        self.image_size = config.get("data", {}).get("image_size", 224)
        self._transform = get_validation_augmentations(self.image_size)

        self.model = AstroClassifier(
            num_classes=num_classes,
            backbone=backbone,
            pretrained=False,
            dropout=dropout,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # Report
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {backbone} | Params: {params:,} | Device: {self.device}")

    @classmethod
    def singleton(cls, **kwargs) -> "ModelManager":
        """Get or create cached singleton instance."""
        if cls._instance is None or kwargs.get("checkpoint_path") != getattr(
            cls._instance, "checkpoint_path", None
        ):
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_singleton(cls):
        """Clear cached instance (useful for testing)."""
        cls._instance = None

    def _warmup(self):
        """Run a dummy forward pass to warm up GPU/cache."""
        if self._warmup_done or self.model is None:
            return

        dummy = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        self._warmup_done = True
        logger.debug("Model warmup complete")

    def _preprocess(self, image: Union[str, Path, np.ndarray]) -> torch.Tensor:
        """Preprocess image to model input tensor."""
        if isinstance(image, (str, Path)):
            img = np.array(Image.open(image).convert("RGB"))
        else:
            img = image
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)

        augmented = self._transform(image=img)["image"]
        return augmented.unsqueeze(0)

    # ------------------------------------------------------------------
    # Single Image Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
    ) -> InferenceResult:
        """
        Predict class for a single image.

        Args:
            image: File path or numpy array [H, W, 3]

        Returns:
            InferenceResult with class name, confidence, probabilities, timing
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._warmup()

        # Preprocess
        t0_preprocess = time.perf_counter()
        input_tensor = self._preprocess(image)
        input_tensor = input_tensor.to(self.device, non_blocking=True)
        t_preprocess = (time.perf_counter() - t0_preprocess) * 1000

        # Inference
        t0_infer = time.perf_counter()

        if self.device.type == "cuda" and hasattr(torch, "amp"):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = self.model(input_tensor)
        else:
            logits = self.model(input_tensor)

        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()

        t_infer = (time.perf_counter() - t0_infer) * 1000

        # Build result
        all_probs = {CLASS_NAMES_DISPLAY[i]: probs[i].item() for i in range(len(CLASS_NAMES_DISPLAY))}
        top_k = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)

        result = InferenceResult(
            class_name=CLASS_NAMES_DISPLAY[pred_idx],
            class_index=pred_idx,
            confidence=confidence,
            all_probabilities=all_probs,
            inference_time_ms=t_infer,
            top_k=top_k,
        )

        logger.info(
            f"Predicted: {result.class_name} ({result.confidence:.2%}) | "
            f"Inference: {result.inference_time_ms:.1f}ms | "
            f"Preprocess: {t_preprocess:.1f}ms"
        )

        return result

    # ------------------------------------------------------------------
    # Batch Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        batch_size: int = 16,
    ) -> List[InferenceResult]:
        """
        Predict classes for multiple images with batching.

        Args:
            images: List of file paths or numpy arrays
            batch_size: Number of images per forward pass

        Returns:
            List of InferenceResult, one per image
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._warmup()

        # Preprocess all images
        t0 = time.perf_counter()
        all_tensors = [self._preprocess(img) for img in images]
        all_tensors = torch.cat(all_tensors, dim=0)
        preprocess_time = (time.perf_counter() - t0) * 1000

        # Process in batches
        all_results = []
        num_batches = (len(images) + batch_size - 1) // batch_size

        t0 = time.perf_counter()

        for i in range(0, len(images), batch_size):
            batch = all_tensors[i:i + batch_size].to(self.device, non_blocking=True)

            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = self.model(batch)
            else:
                logits = self.model(batch)

            probs = torch.softmax(logits, dim=1)
            pred_indices = probs.argmax(dim=1)

            # Build results for this batch
            for j in range(batch.size(0)):
                idx = i + j
                pred_idx = pred_indices[j].item()
                confidence = probs[j, pred_idx].item()
                all_p = {CLASS_NAMES_DISPLAY[k]: probs[j, k].item() for k in range(len(CLASS_NAMES_DISPLAY))}
                top_k = sorted(all_p.items(), key=lambda x: x[1], reverse=True)

                all_results.append(InferenceResult(
                    class_name=CLASS_NAMES_DISPLAY[pred_idx],
                    class_index=pred_idx,
                    confidence=confidence,
                    all_probabilities=all_p,
                    inference_time_ms=0.0,  # Will be filled below
                    top_k=top_k,
                ))

        total_time = (time.perf_counter() - t0) * 1000
        avg_time = total_time / len(images) if images else 0

        # Fill timing
        for r in all_results:
            r.inference_time_ms = avg_time

        logger.info(
            f"Batch inference: {len(images)} images in {total_time:.1f}ms "
            f"({num_batches} batches, avg {avg_time:.1f}ms/image)"
        )

        return all_results

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_classes(self) -> int:
        if self.model is None:
            return 5
        return self.model.num_classes if hasattr(self.model, "num_classes") else 5

    @property
    def class_names(self) -> List[str]:
        return CLASS_NAMES_DISPLAY


# ---------------------------------------------------------------------------
# Convenience Functions (module-level)
# ---------------------------------------------------------------------------

def predict_image(
    image: Union[str, Path, np.ndarray],
    checkpoint_path: str = "checkpoints/best_model.pth",
    device: Optional[str] = None,
) -> InferenceResult:
    """
    One-shot prediction without managing ModelManager.

    Args:
        image: File path or numpy array
        checkpoint_path: Path to model checkpoint
        device: "cuda", "cpu", or None (auto)

    Returns:
        InferenceResult
    """
    manager = ModelManager(checkpoint_path=checkpoint_path, device=device)
    return manager.predict(image)


def predict_batch(
    images: List[Union[str, Path, np.ndarray]],
    checkpoint_path: str = "checkpoints/best_model.pth",
    device: Optional[str] = None,
    batch_size: int = 16,
) -> List[InferenceResult]:
    """One-shot batch prediction."""
    manager = ModelManager(checkpoint_path=checkpoint_path, device=device)
    return manager.predict_batch(images, batch_size=batch_size)


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main():
    """CLI for quick inference tests."""
    import argparse

    parser = argparse.ArgumentParser(description="SCALE x ODYSSEY Inference")
    parser.add_argument("image", type=str, help="Path to image or directory")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    image_path = Path(args.image)

    if image_path.is_file():
        # Single image
        logger.info(f"Inferencing: {image_path}")
        result = predict_image(str(image_path), args.checkpoint, args.device)

        print("\n" + "=" * 50)
        print(f"  Prediction: {result.class_name}")
        print(f"  Confidence: {result.confidence:.2%}")
        print(f"  Time:       {result.inference_time_ms:.1f}ms")
        print("=" * 50)
        print("\nTop 3:")
        for name, prob in result.top_k[:3]:
            print(f"  {name:25s}: {prob:.4f}")

    elif image_path.is_dir():
        # Batch
        image_files = [
            str(f) for f in image_path.glob("*")
            if f.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        if not image_files:
            logger.error(f"No images found in {image_path}")
            return

        logger.info(f"Found {len(image_files)} images in {image_path}")
        results = predict_batch(image_files, args.checkpoint, args.device, args.batch_size)

        print("\n" + "=" * 60)
        print(f"{'Image':<30s} {'Prediction':<20s} {'Confidence':>10s}")
        print("-" * 60)
        for path, res in zip(image_files, results):
            fname = Path(path).name[:28]
            print(f"{fname:<30s} {res.class_name:<20s} {res.confidence:>9.1%}")
        print("=" * 60)

    else:
        logger.error(f"Path not found: {image_path}")


if __name__ == "__main__":
    main()
