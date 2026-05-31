import pytest
from src.dataset import AstroDataset, get_train_transforms, get_val_transforms


def test_transforms():
    t = get_train_transforms(224)
    assert t is not None
    v = get_val_transforms(224)
    assert v is not None
