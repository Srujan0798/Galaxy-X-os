"""Unit test for src/inference.py ModelManager model-not-found behavior."""

import pytest

from src.inference import ModelManager


def test_model_manager_raises_on_missing_checkpoint(tmp_path):
    missing = tmp_path / "does_not_exist.pth"
    with pytest.raises(FileNotFoundError):
        ModelManager(checkpoint_path=str(missing))