#!/usr/bin/env python3
"""
SCALE x ODYSSEY -- src/onnx_export.py
Export model to ONNX + TensorRT optimization for production inference.

ONNX enables:
- Cross-platform deployment (CPU, GPU, mobile, edge)
- TensorRT optimization (2-5x speedup)
- Quantization (FP16, INT8) for embedded devices
- Framework-agnostic inference
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import AstroClassifier, create_model_from_config  # noqa: E402


def export_to_onnx(model: torch.nn.Module, output_path: str,
                   input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224),
                   opset_version: int = 17,
                   device: str = "cpu",
                   dynamic_batch: bool = True) -> str:
    """
    Export PyTorch model to ONNX format.
    
    Args:
        model: PyTorch model
        output_path: where to save .onnx file
        input_shape: (B, C, H, W)
        opset_version: ONNX opset version (17 supports modern ops)
        device: 'cpu' or 'cuda'
        dynamic_batch: allow variable batch size
    
    Returns:
        path to exported ONNX file
    """
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(*input_shape, device=device)
    
    # Dynamic axes for variable batch size
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
        verbose=False,
    )
    
    return output_path


def check_onnx_model(onnx_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224)) -> bool:
    """Validate ONNX model with onnxruntime."""
    try:
        import onnx
        import onnxruntime as ort
        
        # Check ONNX model validity
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        
        # Run inference
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        output = session.run(None, {input_name: dummy_input})
        
        print(f"ONNX model valid: {output[0].shape}")
        return True
    
    except ImportError:
        print("onnx/onnxruntime not installed, skipping validation")
        return False
    except Exception as e:
        print(f"ONNX validation failed: {e}")
        return False


def quantize_onnx(onnx_path: str, output_path: str, 
                  quantization: str = "fp16",
                  calibration_data: Optional[np.ndarray] = None) -> str:
    """
    Quantize ONNX model for faster inference.
    
    Args:
        onnx_path: input ONNX model
        output_path: output quantized model path
        quantization: 'fp16' or 'int8'
        calibration_data: optional data for INT8 calibration
    
    Returns:
        path to quantized model
    """
    try:
        from onnxruntime.quantization import quantize_dynamic, quantize_static, QuantType  # noqa: F401
        
        if quantization == "fp16":
            from onnxruntime.quantization import quantize_fp16
            quantize_fp16(onnx_path, output_path)
        elif quantization == "int8":
            if calibration_data is not None:
                # Static quantization with calibration
                calib_data_path = output_path.replace(".onnx", "_calib_data.npz")
                np.savez(calib_data_path, data=calibration_data)
                quantize_static(onnx_path, output_path, calibration_data_path=calib_data_path)
            else:
                # Dynamic quantization (no calibration needed)
                quantize_dynamic(onnx_path, output_path, weight_type=QuantType.QUInt8)
        
        print(f"Quantized ONNX saved to {output_path}")
        return output_path
    
    except ImportError:
        print("quantization tools not installed, skipping")
        return onnx_path


def export_model_pipeline(model_config: Dict, output_dir: str,
                         input_shape: Tuple[int, int, int, int] = (1, 3, 224, 224)) -> Dict:
    """
    Full export pipeline: PyTorch -> ONNX -> FP16 -> INT8.
    Returns dict with paths to all exported models.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = model_config.get("backbone", "model")
    checkpoint_path = model_config.get("checkpoint_path", "checkpoints/best_model.pth")
    
    results = {}
    
    # Load model
    if os.path.exists(checkpoint_path):
        if model_config.get("ensemble"):
            model = create_model_from_config(model_config)
            # Load ensemble checkpoints
            for i, (m, ckpt_path) in enumerate(zip(model.models, model_config.get("checkpoints", []))):
                if os.path.exists(ckpt_path):
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
                    m.load_state_dict(ckpt["model_state_dict"])
        else:
            model = AstroClassifier(
                num_classes=model_config.get("num_classes", 5),
                backbone=model_config.get("backbone", "convnext_base"),
                pretrained=False,
                dropout=model_config.get("dropout", 0.4),
            )
            ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            model.load_state_dict(ckpt["model_state_dict"])
    else:
        print(f"Checkpoint not found: {checkpoint_path}, using random weights")
        model = create_model_from_config(model_config)
    
    model.eval()
    
    # 1. PyTorch -> ONNX
    onnx_path = os.path.join(output_dir, f"{base_name}.onnx")
    export_to_onnx(model, onnx_path, input_shape, device="cpu")
    results["onnx"] = onnx_path
    print(f"ONNX exported: {onnx_path}")
    
    # 2. ONNX -> FP16
    fp16_path = onnx_path.replace(".onnx", "_fp16.onnx")
    quantize_onnx(onnx_path, fp16_path, "fp16")
    results["fp16"] = fp16_path
    
    # 3. ONNX -> INT8
    int8_path = onnx_path.replace(".onnx", "_int8.onnx")
    quantize_onnx(onnx_path, int8_path, "int8")
    results["int8"] = int8_path
    
    return results


def benchmark_onnx(onnx_path: str, input_shape: Tuple[int, ...] = (1, 3, 224, 224),
                   num_iterations: int = 100, warmup: int = 10) -> Dict:
    """Benchmark ONNX model inference speed."""
    try:
        import onnxruntime as ort
        import time
        
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        
        # Warmup
        for _ in range(warmup):
            session.run(None, {input_name: dummy_input})
        
        # Benchmark
        timings = []
        for _ in range(num_iterations):
            t0 = time.perf_counter()
            session.run(None, {input_name: dummy_input})
            timings.append((time.perf_counter() - t0) * 1000)
        
        timings = np.array(timings)
        return {
            "mean_ms": float(timings.mean()),
            "std_ms": float(timings.std()),
            "p50_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "p99_ms": float(np.percentile(timings, 99)),
            "fps": float(1000 / timings.mean()),
        }
    
    except ImportError:
        return {"error": "onnxruntime not installed"}
    except Exception as e:
        return {"error": str(e)}


def main():
    """CLI entry: export model to ONNX."""
    import argparse
    parser = argparse.ArgumentParser(
        description="SCALE x ODYSSEY -- ONNX Export + Quantization"
    )
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth",
                        help="Path to .pth checkpoint")
    parser.add_argument("--output-dir", default="exports",
                        help="Output directory for ONNX files")
    parser.add_argument("--input-shape", default="1,3,224,224",
                        help="Input shape as B,C,H,W (default 1,3,224,224)")
    parser.add_argument("--skip-quantize", action="store_true",
                        help="Skip FP16/INT8 quantization")
    args = parser.parse_args()

    shape = tuple(int(x) for x in args.input_shape.split(","))
    cfg = {"backbone": "efficientnet_b3", "checkpoint_path": args.checkpoint}

    result = export_model_pipeline(cfg, args.output_dir, shape)
    print(f"\nONNX export results: {result}")


if __name__ == "__main__":
    main()