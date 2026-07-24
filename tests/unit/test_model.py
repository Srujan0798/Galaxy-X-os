import torch
import pytest
from src.model import AstroClassifier


def test_model_forward():
    # tiny backbone, no pretrained download -- fast on CPU.
    # Use a 224x224 input (the model's expected size) + eval mode to avoid
    # BatchNorm complaining on a 1x1 spatial feature map.
    model = AstroClassifier(num_classes=5, backbone="resnet18",
                            pretrained=False, dropout=0.0)
    model.eval()
    with torch.no_grad():
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
    assert out.shape == (1, 5)


def test_freeze_backbone():
    model = AstroClassifier(num_classes=5, backbone="resnet18",
                            pretrained=False)
    model.freeze_backbone()
    for p in model.backbone.parameters():
        assert not p.requires_grad
