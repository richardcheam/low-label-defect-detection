from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    output_dim = 4096

    def __init__(self, input_channels: int = 1) -> None:
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(p=0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.layer1(inputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.pool(outputs)
        return outputs.view(outputs.size(0), -1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int = Encoder.output_dim, output_dim: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.layers(features)


class LinearProbe(nn.Module):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(encoder.output_dim, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


class MLPProbe(nn.Module):
    def __init__(self, encoder: Encoder) -> None:
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


class EncoderClassifier(nn.Module):
    def __init__(
        self,
        encoder: Encoder | None = None,
        hidden_dim: int = 128,
        num_classes: int = 10,
        input_channels: int = 1,
    ) -> None:
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder(input_channels=input_channels)
        self.classifier = nn.Sequential(
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encoder(inputs))


class SemiSupervisedCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding="same")
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(128 * 2 * 2, 192)
        self.batchnorm4 = nn.BatchNorm1d(192)
        self.fc2 = nn.Linear(192, 64)
        self.batchnorm5 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pool1(torch.relu(self.batchnorm1(self.conv1(inputs))))
        outputs = self.pool2(torch.relu(self.batchnorm2(self.conv2(outputs))))
        outputs = self.pool3(torch.relu(self.batchnorm3(self.conv3(outputs))))
        outputs = torch.flatten(outputs, 1)
        outputs = torch.relu(self.batchnorm4(self.fc1(outputs)))
        outputs = torch.relu(self.batchnorm5(self.fc2(outputs)))
        return self.fc3(outputs)
