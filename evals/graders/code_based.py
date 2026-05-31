# Eval Graders — Code Based

import pytest
import torch


def test_config_load():
    from src.utils import load_config
    cfg = load_config("config/config.yaml")
    assert "data" in cfg
    assert "model" in cfg
    assert "training" in cfg
    assert "paths" in cfg


def test_model_forward():
    from src.model import AstroClassifier
    model = AstroClassifier(num_classes=5, backbone="efficientnet_b3")
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 5)


def test_dataset_load():
    # Requires data to exist
    pass


def test_inference_speed():
    # Requires checkpoint
    pass


def test_app_launch():
    # Requires streamlit
    pass
