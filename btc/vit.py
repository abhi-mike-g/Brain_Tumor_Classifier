"""Compact Vision Transformer for brain MRI image classification."""
from __future__ import annotations

import torch
import torch.nn as nn


class VisionTransformer(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        grid_size = image_size // patch_size
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position = nn.Parameter(torch.zeros(1, grid_size * grid_size + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.position, std=0.02)
        nn.init.trunc_normal_(self.class_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1) + self.position
        encoded = self.encoder(tokens)
        return self.head(self.norm(encoded[:, 0]))


def build_model(
    num_classes: int = 4,
    image_size: int = 224,
    patch_size: int = 16,
    embed_dim: int = 192,
    depth: int = 4,
    num_heads: int = 3,
    dropout: float = 0.1,
    **kwargs,
) -> nn.Module:
    return VisionTransformer(
        num_classes=num_classes,
        image_size=image_size,
        patch_size=patch_size,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        dropout=dropout,
    )
