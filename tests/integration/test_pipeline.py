"""
Integration test: prepare_data.py can assemble a tiny 5-class dataset.

This runs the real pipeline code but with --per-class=10 so it completes
quickly on CPU. It verifies the output directory structure, the manifest,
and the MD5 leak check pass without raising.
"""

import subprocess
import sys
from pathlib import Path


def test_prepare_data_pipeline():
    result = subprocess.run(
        [sys.executable, "src/prepare_data.py", "--per-class", "10", "--seed", "7"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    processed = Path("data/processed")
    assert (processed / "DATA_MANIFEST.json").exists()
    assert (processed / "class_weights.json").exists()
    for split in ("train", "val", "test"):
        for cls in ("spiral_galaxy", "elliptical_galaxy", "nebula", "star_cluster", "planetary"):
            class_dir = processed / split / cls
            assert class_dir.exists(), f"missing {class_dir}"
            assert len(list(class_dir.glob("*.png"))) > 0, f"no images in {class_dir}"
