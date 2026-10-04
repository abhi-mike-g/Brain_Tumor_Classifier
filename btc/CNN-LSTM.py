"""CNN-LSTM (Member 3 / Vansh): CNN feature extractor followed by LSTM for 4-class brain tumor classification.""""

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class CNNLSTM(nn.Module):
    """CNN-LSTM model for brain tumor classification."""

    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        lstm_hidden_size: int = 256,
        lstm_layers: int = 1,
        dropout: float = 0.3,
    ):
        super().__init__()
        backbone = resnet18(
            weights=ResNet18_Weights.IMAGENET1K_V1
            if pretrained else None
        )
        self.cnn = nn.Sequential(
            *list(backbone.children())[:-2]
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Linear(
            lstm_hidden_size,
            num_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cnn(x)

        batch_size, channels, height, width = x.shape

        x = x.permute(0, 2, 3, 1)

        x = x.reshape(
            batch_size,
            height * width,
            channels
        )
        x, _ = self.lstm(x)

        # Take the last sequence output
        x = x[:, -1, :]

        # Classification
        x = self.dropout(x)

        return self.fc(x)

def build_model(
    num_classes: int = 4,
    pretrained: bool = True,
    dropout: float = 0.3,
    lstm_hidden_size: int = 256,
    lstm_layers: int = 1,
    **kwargs,
) -> nn.Module:

    return CNNLSTM(
        num_classes=num_classes,
        pretrained=pretrained,
        lstm_hidden_size=lstm_hidden_size,
        lstm_layers=lstm_layers,
        dropout=dropout,
    )
