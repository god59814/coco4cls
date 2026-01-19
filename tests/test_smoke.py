from __future__ import annotations
import torch
from coco4cls.model import build_model, build_preprocess

def test_model_forward_smoke():
    model = build_model(backbone="efficientnet_b0", num_classes=4, pretrained=False)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        y = model(x)
    assert y.shape == (2, 4)

def test_preprocess_smoke():
    tfm = build_preprocess(224)
    assert tfm is not None
