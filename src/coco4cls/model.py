from __future__ import annotations
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

def build_preprocess(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

def _replace_classifier(backbone: nn.Module, backbone_name: str, num_classes: int) -> nn.Module:
    if backbone_name.startswith("resnet"):
        in_features = backbone.fc.in_features
        backbone.fc = nn.Linear(in_features, num_classes)
        return backbone
    if backbone_name.startswith("efficientnet"):
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        return backbone
    if backbone_name.startswith("mobilenet_v3"):
        in_features = backbone.classifier[-1].in_features
        backbone.classifier[-1] = nn.Linear(in_features, num_classes)
        return backbone
    raise ValueError(f"Unsupported backbone: {backbone_name}")

def build_model(backbone: str, num_classes: int, freeze_backbone: bool = False, pretrained: bool = True) -> nn.Module:
    if backbone == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    elif backbone == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
    elif backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
    elif backbone == "mobilenet_v3_small":
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")
    m = _replace_classifier(m, backbone, num_classes)

    if freeze_backbone:
        for name, p in m.named_parameters():
            if "classifier" in name or name.endswith(".fc.weight") or name.endswith(".fc.bias"):
                p.requires_grad = True
            else:
                p.requires_grad = False
    return m

@torch.inference_mode()
def predict_image(model: nn.Module, image: Image.Image, preprocess: transforms.Compose, device: str, categories: List[str], topk: int = 4) -> Dict[str, Any]:
    x = preprocess(image).unsqueeze(0).to(device)
    logits = model(x)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    k = min(topk, probs.numel())
    values, indices = torch.topk(probs, k=k)
    items: List[Dict[str, Any]] = []
    for v, i in zip(values.tolist(), indices.tolist()):
        items.append({"class_id": int(i), "class_name": categories[int(i)], "prob": float(v)})
    return {"topk": items, "pred": items[0] if items else None}
