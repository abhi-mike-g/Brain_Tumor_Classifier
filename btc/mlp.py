"""MLP Baseline (Member 1 / Varun Naik): Fully connected network over flattened 224x224 MRI images."""
from __future__ import annotations

from typing import Sequence
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron baseline with LayerNorm and Dropout."""

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten(start_dim=1)]
        prev_dim = in_features
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(
    num_classes: int = 4,
    in_size: int = 224,
    hidden_dims: Sequence[int] = (512, 256, 64),
    dropout: float = 0.3,
    **kwargs,
) -> nn.Module:
    """Build the MLP model. Safely ignores architecture-specific kwargs like pretrained."""
    in_features = 3 * in_size * in_size
    return MLP(
        in_features=in_features,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=dropout,
    )
