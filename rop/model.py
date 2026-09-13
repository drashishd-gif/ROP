"""A small CNN for ROP (Retinopathy of Prematurity) fundus-image
classification, built and trained from scratch.

No pretrained weights and no external inference API are used anywhere in
this project: every parameter in this network is learned from the images
you train it on (e.g. a Kaggle ROP dataset laid out with rop/dataset.py).
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, pool: bool = True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.pool(x)
        return x


class ROPNet(nn.Module):
    """Custom CNN classifier for retinal fundus images.

    Input:  (B, 3, H, W) RGB fundus image, normalized to [0, 1] and then
            standardized with ImageNet-style mean/std (see rop/dataset.py).
    Output: (B, num_classes) raw logits.
    """

    def __init__(self, num_classes: int = 2, in_channels: int = 3, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_channels, 32),   # -> H/2
            ConvBlock(32, 64),            # -> H/4
            ConvBlock(64, 128),           # -> H/8
            ConvBlock(128, 256),          # -> H/16
            ConvBlock(256, 256, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the last conv feature map, used for Grad-CAM style
        visualization of which region of the image drove the prediction."""
        return self.features(x)


def build_model(num_classes: int, dropout: float = 0.4) -> ROPNet:
    return ROPNet(num_classes=num_classes, dropout=dropout)
