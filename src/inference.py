#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/inference.py
Fast Inference Module

Single-image and batch inference with timing.
ModelManager singleton for web app caching.
Target: <15ms per image on GPU, <5s total.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
from dataclasses import dataclass

# When this file is loaded as ``src.inference`` (test runner) the absolute
# ``from model import ...`` below fails because ``src/`` is not on sys.path.
# Prepend this file's directory so the siblings resolve either way.
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from model import AstroClassifier  # noqa: E402
from dataset import CLASS_NAMES_DISPLAY, get_val_transforms  # noqa: E402
from utils import get_device  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Structured inference result."""
    class_name: str
    class_index: int
    confidence: float
    all_probabilities: Dict[str, float]
    inference_time_ms: float
    top_k: List[Tuple[str, float]]

    def to_dict(self) -> Dict:
        return {
            "class_name": self.class_name,
            "confidence": self.confidence,
            "inference_time_ms": round(self.inference_time_ms, 2),
            "top_3": [(n, round(float(p), 4)) for n, p in self.top_k[:3]],
        }


class ModelManager:
    """Manages model loading with singleton caching."""

    _instance: Optional["ModelManager"] = None

    def __init__(self, checkpoint_path: str = "checkpoints/best_model.pth",
                 device: Optional[str] = None):
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device) if device else get_device()
        self.model: Optional[nn.Module] = None
        self.image_size: int = 224
        self._transform = get_val_transforms(self.image_size)
        self._warmup_done = False
        self._load_model()

    def _load_model(self):
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
        cfg = checkpoint.get("config", {})
        mc = cfg.get("model", {})
        self.image_size = cfg.get("data", {}).get("image_size", 224)
        self._transform = get_val_transforms(self.image_size)

        self.model = AstroClassifier(mc.get("num_classes", 5), mc.get("backbone", "efficientnet_b3"),
                                     pretrained=False, dropout=mc.get("dropout", 0.4))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Model loaded: {mc.get('backbone', 'efficientnet_b3')} | "
                    f"Params: {params:,} | Device: {self.device}")

    @classmethod
    def singleton(cls, **kwargs) -> "ModelManager":
        key = tuple(sorted(kwargs.items()))
        if cls._instance is None or key != getattr(cls._instance, "_cache_key", None):
            cls._instance = cls(**kwargs)
            cls._instance._cache_key = key
        return cls._instance

    def _warmup(self):
        if self._warmup_done or self.model is None:
            return
        dummy = torch.randn(1, 3, self.image_size, self.image_size, device=self.device)
        with torch.no_grad():
            _ = self.model(dummy)
        self._warmup_done = True

    def _preprocess(self, image: Union[str, Path, np.ndarray]) -> torch.Tensor:
        if isinstance(image, (str, Path)):
            img = np.array(Image.open(image).convert("RGB"))
        else:
            img = image if image.ndim == 3 else np.stack([image] * 3, axis=-1)
        return self._transform(image=img)["image"].unsqueeze(0)

    @torch.no_grad()
    def predict(self, image: Union[str, Path, np.ndarray]) -> InferenceResult:
        """Predict class for a single image."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        self._warmup()

        input_tensor = self._preprocess(image).to(self.device, non_blocking=True)
        t0 = time.perf_counter()

        if self.device.type in ("cuda", "mps"):
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                logits = self.model(input_tensor)
        else:
            logits = self.model(input_tensor)

        probs = torch.softmax(logits, dim=1)[0]
        pred_idx = probs.argmax().item()
        confidence = probs[pred_idx].item()
        t_infer = (time.perf_counter() - t0) * 1000

        all_probs = {CLASS_NAMES_DISPLAY[i]: probs[i].item() for i in range(len(CLASS_NAMES_DISPLAY))}
        result = InferenceResult(
            class_name=CLASS_NAMES_DISPLAY[pred_idx], class_index=pred_idx,
            confidence=confidence, all_probabilities=all_probs,
            inference_time_ms=t_infer, top_k=sorted(all_probs.items(), key=lambda x: x[1], reverse=True),
        )
        logger.info(f"Predicted: {result.class_name} ({result.confidence:.2%}) | "
                    f"Time: {result.inference_time_ms:.1f}ms")
        return result

    @torch.no_grad()
    def predict_batch(self, images: List[Union[str, Path, np.ndarray]], batch_size: int = 16) -> List[InferenceResult]:
        """Predict classes for multiple images."""
        if self.model is None:
            raise RuntimeError("Model not loaded")
        self._warmup()

        all_tensors = torch.cat([self._preprocess(img) for img in images], dim=0)
        all_results = []

        t0 = time.perf_counter()
        for i in range(0, len(images), batch_size):
            batch = all_tensors[i:i + batch_size].to(self.device, non_blocking=True)
            if self.device.type in ("cuda", "mps"):
                with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                    logits = self.model(batch)
            else:
                logits = self.model(batch)
            probs = torch.softmax(logits, dim=1)
            for j in range(batch.size(0)):
                pred_idx = probs[j].argmax().item()
                all_p = {CLASS_NAMES_DISPLAY[k]: probs[j, k].item() for k in range(len(CLASS_NAMES_DISPLAY))}
                all_results.append(InferenceResult(
                    class_name=CLASS_NAMES_DISPLAY[pred_idx], class_index=pred_idx,
                    confidence=probs[j, pred_idx].item(), all_probabilities=all_p,
                    inference_time_ms=0.0,
                    top_k=sorted(all_p.items(), key=lambda x: x[1], reverse=True),
                ))

        total_time = (time.perf_counter() - t0) * 1000
        avg_time = total_time / len(images) if images else 0
        for r in all_results:
            r.inference_time_ms = avg_time
        logger.info(f"Batch: {len(images)} images in {total_time:.1f}ms (avg {avg_time:.1f}ms/image)")
        return all_results


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------

def predict_image(image: Union[str, Path, np.ndarray],
                  checkpoint_path: str = "checkpoints/best_model.pth",
                  device: Optional[str] = None) -> InferenceResult:
    return ModelManager(checkpoint_path=checkpoint_path, device=device).predict(image)


def predict_batch(images: List[Union[str, Path, np.ndarray]],
                  checkpoint_path: str = "checkpoints/best_model.pth",
                  device: Optional[str] = None, batch_size: int = 16) -> List[InferenceResult]:
    return ModelManager(checkpoint_path=checkpoint_path, device=device).predict_batch(images, batch_size)


# CLI
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SCALE x ODYSSEY Inference")
    parser.add_argument("image", type=str, help="Path to image or directory")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--device", default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    image_path = Path(args.image)
    if image_path.is_file():
        result = predict_image(str(image_path), args.checkpoint, args.device)
        print(f"\n{'='*50}\n  Prediction: {result.class_name}\n  Confidence: {result.confidence:.2%}\n  "
              f"Time: {result.inference_time_ms:.1f}ms\n{'='*50}")
        print("\nTop 3:")
        for name, prob in result.top_k[:3]:
            print(f"  {name:25s}: {prob:.4f}")
    elif image_path.is_dir():
        files = sorted(
            str(f)
            for pat in ("*.png", "*.jpg", "*.jpeg")
            for f in image_path.rglob(pat)
        )
        if not files:
            logger.error("No images found (recursively) in %s", image_path)
            raise SystemExit(1)
        results = predict_batch(files, args.checkpoint, args.device, args.batch_size)
        print(f"\n{'='*60}\n{'Image':<30s} {'Prediction':<20s} {'Conf':>8s}\n{'-'*60}")
        for p, r in zip(files, results):
            print(f"{Path(p).name[:28]:<30s} {r.class_name:<20s} {r.confidence:>7.1%}")
        print("=" * 60)
    else:
        logger.error(f"Path not found: {image_path}")
        raise SystemExit(1)
