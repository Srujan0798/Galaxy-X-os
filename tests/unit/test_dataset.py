"""Unit test for src/dataset.py: transforms are constructible and correct shape."""

import numpy as np

from src.dataset import AstroDataset, get_train_transforms, get_val_transforms

np.random.seed(42)


def test_train_transforms_produce_tensor():
    t = get_train_transforms(224)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out = t(image=img)["image"]
    # ToTensorV2 returns a CxHxW float tensor.
    assert tuple(out.shape[-2:]) == (224, 224)


def test_val_transforms_produce_tensor():
    t = get_val_transforms(224)
    img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    out = t(image=img)["image"]
    assert tuple(out.shape[-2:]) == (224, 224)


def test_astro_dataset_handles_missing_split_dir(tmp_path):
    # An absent split dir should yield an empty dataset (not raise).
    ds = AstroDataset(str(tmp_path), split="nonexistent")
    assert len(ds) == 0
    assert ds.labels == []