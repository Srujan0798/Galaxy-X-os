"""Unit tests for src/onnx_export.py — CLI and export helpers."""

import subprocess
import sys


def test_onnx_export_help():
    result = subprocess.run(
        [sys.executable, "src/onnx_export.py", "--help"],
        capture_output=True, text=True, check=True,
    )
    assert "ONNX Export" in result.stdout


def test_onnx_export_module_importable():
    from src.onnx_export import export_to_onnx
    assert callable(export_to_onnx)


def test_export_functions_exist():
    from src.onnx_export import (
        export_to_onnx,
        check_onnx_model,
        quantize_onnx,
        export_model_pipeline,
        benchmark_onnx,
    )
    assert callable(export_to_onnx)
    assert callable(check_onnx_model)
    assert callable(quantize_onnx)
    assert callable(export_model_pipeline)
    assert callable(benchmark_onnx)
