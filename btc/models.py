"""Model construction shared by training and evaluation."""
from __future__ import annotations

import torch.nn as nn


def build_model(cfg: dict) -> nn.Module:
    name = cfg["model_name"]
    classes = len(cfg["classes"])
    if name == "mlp":
        from .mlp import build_model as build

        return build(
            num_classes=classes,
            in_size=cfg["image"]["size"],
            downsample_size=cfg.get("downsample_size", 56),
            hidden_dims=tuple(cfg.get("hidden_dims", (512, 256, 64))),
            dropout=cfg.get("dropout", 0.3),
        )
    if name == "resnet18":
        from .resnet18 import build_model as build

        return build(
            num_classes=classes,
            pretrained=cfg.get("pretrained", True),
            dropout=cfg.get("dropout", 0.2),
        )
    if name == "cnn_lstm":
        from .cnn_lstm import build_model as build

        return build(
            num_classes=classes,
            lstm_hidden_size=cfg.get("lstm_hidden_size", 128),
            dropout=cfg.get("dropout", 0.3),
        )
    if name == "vit":
        from .vit import build_model as build

        return build(
            num_classes=classes,
            image_size=cfg["image"]["size"],
            patch_size=cfg.get("patch_size", 16),
            embed_dim=cfg.get("embed_dim", 192),
            depth=cfg.get("depth", 4),
            num_heads=cfg.get("num_heads", 3),
            dropout=cfg.get("dropout", 0.1),
        )
    raise ValueError(f"Unknown model_name in config: {name!r}")
