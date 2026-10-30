"""CNN-LSTM classifier that treats the CNN feature-map rows as a sequence."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int = 4,
        lstm_hidden_size: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_pool = nn.AvgPool2d(2)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=lstm_hidden_size,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(self.input_pool(x))
        batch, channels, height, width = features.shape
        sequence = features.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        sequence, _ = self.lstm(sequence)
        return self.classifier(self.dropout(sequence[:, -1]))


def build_model(
    num_classes: int = 4,
    lstm_hidden_size: int = 128,
    dropout: float = 0.3,
    **kwargs,
) -> nn.Module:
    return CNNLSTM(
        num_classes=num_classes,
        lstm_hidden_size=lstm_hidden_size,
        dropout=dropout,
    )
