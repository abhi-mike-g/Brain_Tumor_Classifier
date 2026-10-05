"""CNN-LSTM (Member 3 / Abhidutta Mukund Giri): convolutional features read as a sequence by an LSTM."""
from __future__ import annotations

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """Five conv blocks reduce a 224x224 MRI to a 7x7 grid, then an LSTM reads that grid."""

    def __init__(
        self,
        num_classes: int = 4,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)
        batch_size, channels, height, width = x.shape
        x = x.permute(0, 2, 3, 1).reshape(batch_size, height * width, channels)
        x, _ = self.lstm(x)
        x = self.dropout(x[:, -1, :])
        return self.fc(x)


def build_model(
    num_classes: int = 4,
    dropout: float = 0.3,
    lstm_hidden_size: int = 256,
    lstm_layers: int = 1,
    **kwargs,
) -> nn.Module:
    """Build CNN-LSTM. Ignores kwargs such as pretrained (this model is trained from scratch)."""
    return CNNLSTM(
        num_classes=num_classes,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )
