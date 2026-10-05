"""Build whichever architecture a config names, so train and evaluate stay in lockstep."""
from __future__ import annotations

import torch.nn as nn

MODEL_CHOICES = ("resnet18", "mlp", "cnn_lstm", "vit")
LABELS = {
    "resnet18": "ResNet18",
    "mlp": "MLP",
    "cnn_lstm": "CNN-LSTM",
    "vit": "ViT",
}


def normalize_name(name: str) -> str:
    key = name.lower().replace("-", "_")
    if key in {"cnnlstm", "cnn_lstm"}:
        return "cnn_lstm"
    if key not in MODEL_CHOICES:
        raise ValueError(f"unknown model {name!r}; expected one of {MODEL_CHOICES}")
    return key


def build_model(cfg: dict, *, pretrained: bool | None = None) -> nn.Module:
    name = normalize_name(cfg.get("model_name", "resnet18"))
    num_classes = len(cfg["classes"])
    if pretrained is None:
        pretrained = bool(cfg.get("pretrained", False))
    dropout = float(cfg.get("dropout", 0.2))
    if name == "mlp":
        from .mlp import build_model as build_mlp

        return build_mlp(
            num_classes=num_classes,
            in_size=cfg["image"]["size"],
            hidden_dims=tuple(cfg.get("hidden_dims", (512, 256, 64))),
            dropout=float(cfg.get("dropout", 0.3)),
        )
    if name == "cnn_lstm":
        from .cnn_lstm import build_model as build_cnn_lstm

        return build_cnn_lstm(
            num_classes=num_classes,
            dropout=dropout,
            lstm_hidden_size=int(cfg.get("lstm_hidden_size", 256)),
            lstm_layers=int(cfg.get("lstm_layers", 1)),
        )
    if name == "vit":
        from .vit import build_model as build_vit

        return build_vit(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
    from .resnet18 import build_model as build_resnet18

    return build_resnet18(num_classes=num_classes, pretrained=pretrained, dropout=dropout)
