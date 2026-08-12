import torch

from src.model import ViT, build_model
from src.config import ModelConfig


def test_vit_output_shape():
    num_patches = (32 // 4) ** 2
    model = ViT(patch_size=4, num_patches=num_patches, num_classes=100)
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    assert out.shape == (2, 100)


def test_build_model_from_config():
    config = ModelConfig(num_classes=10)
    num_patches = (32 // config.patch_size) ** 2
    model = build_model(config, num_patches)
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


def test_forward_is_differentiable():
    num_patches = (32 // 4) ** 2
    model = ViT(patch_size=4, num_patches=num_patches, num_classes=5)
    out = model(torch.randn(1, 3, 32, 32))
    out.sum().backward()
    assert model.class_token.grad is not None
