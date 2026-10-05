"""Vision Transformer (Member 4 / Vansh Vakharia): torchvision ViT-B/32 with a 4-class head."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import ViT_B_32_Weights, vit_b_32


def build_model(num_classes: int = 4, pretrained: bool = True, dropout: float = 0.1, **kwargs) -> nn.Module:
    """ImageNet-initialised ViT-B/32 (patch size 32, 224 input -> 7x7 patches).

    The classification head is replaced for the four MRI classes. `dropout` is the
    encoder dropout used by torchvision's VisionTransformer.
    """
    weights = ViT_B_32_Weights.IMAGENET1K_V1 if pretrained else None
    model = vit_b_32(weights=weights, dropout=dropout)
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, num_classes)
    return model
