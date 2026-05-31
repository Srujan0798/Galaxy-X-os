import torch
import pytest
from src.model import AstroClassifier


def test_model_forward():
    model = AstroClassifier(num_classes=5, backbone="efficientnet_b3")
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 5)


def test_freeze_backbone():
    model = AstroClassifier(num_classes=5)
    model.freeze_backbone()
    for p in model.backbone.parameters():
        assert not p.requires_grad
