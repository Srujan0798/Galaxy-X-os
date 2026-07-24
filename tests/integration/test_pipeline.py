"""
Integration test: prepare_data.py can assemble a tiny 5-class dataset.

Runs the real pipeline code with --per-class=10 into a TEMP directory so it
NEVER overwrites the committed data/processed/ manifest. Verifies the output
directory structure, the manifest, the class-weights file, and the split layout.
"""

import subprocess
import sys
from pathlib import Path


def test_prepare_data_pipeline(tmp_path: Path):
    out_dir = tmp_path / "processed"
    result = subprocess.run(
        [sys.executable, "src/prepare_data.py",
         "--per-class", "10", "--seed", "7",
         "--output-dir", str(out_dir)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    assert (out_dir / "DATA_MANIFEST.json").exists()
    assert (out_dir / "class_weights.json").exists()
    for split in ("train", "val", "test"):
        for cls in ("spiral_galaxy", "elliptical_galaxy", "nebula", "star_cluster", "planetary"):
            class_dir = out_dir / split / cls
            assert class_dir.exists(), f"missing {class_dir}"
            assert len(list(class_dir.glob("*.png"))) > 0, f"no images in {class_dir}"