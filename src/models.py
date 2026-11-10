"""Model definitions: the scratch baseline CNN and the VGG16 transfer-learning model."""
from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class BaselineCNN(nn.Module):
    """4 x [Conv3x3-BN-ReLU-MaxPool] -> FC128 -> Dropout -> FC4, for 64x64 inputs.

    Spatial size: 64 -> 32 -> 16 -> 8 -> 4, so the flattened vector is 128*4*4 = 2048.
    """

    def __init__(self, channels=(32, 64, 128, 128), fc=128, dropout=0.5, n_classes=4, input_size=64):
        super().__init__()
        layers, c_in = [], 3
        for c in channels:
            layers += [nn.Conv2d(c_in, c, 3, padding=1, bias=False), nn.BatchNorm2d(c),
                       nn.ReLU(inplace=True), nn.MaxPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        spatial = input_size // (2 ** len(channels))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(c_in * spatial * spatial, fc), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fc, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


class VGGHead(nn.Module):
    """Replacement classifier: Flatten(512*4*4 = 8192) -> Dense256 + ReLU -> Dropout -> Dense4."""

    def __init__(self, in_features=8192, hidden=256, dropout=0.5, n_classes=4):
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(in_features, hidden), nn.ReLU(inplace=True),
                                 nn.Dropout(dropout), nn.Linear(hidden, n_classes))

    def forward(self, x):
        return self.net(x)

    def dense_weights(self):
        """Kernels of both Dense layers (targets of the L2 penalty)."""
        return [m.weight for m in self.net if isinstance(m, nn.Linear)]


def load_vgg16_features(pretrained: bool = True) -> nn.Sequential:
    """torchvision VGG16 convolutional part (31 layers: blocks 1-5)."""
    weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    return vgg16(weights=weights).features


class VGG16Classifier(nn.Module):
    """Full inference model: ImageNet normalisation -> VGG16 features -> new head.

    ``forward`` expects float images in [0, 1], shape (N, 3, 128, 128).
    Block indices: features[:24] = blocks 1-4, features[24:31] = block 5.
    """

    def __init__(self, hidden=256, dropout=0.5, n_classes=4, pretrained=True, input_size=128):
        super().__init__()
        self.features = load_vgg16_features(pretrained)
        spatial = input_size // 32
        self.head = VGGHead(512 * spatial * spatial, hidden, dropout, n_classes)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.head(self.features(x))


class Block5Head(nn.Module):
    """Stage-2 trainable part: block 5 (features[24:31]) + head, fed with cached block-4 maps."""

    def __init__(self, block5: nn.Sequential, head: VGGHead):
        super().__init__()
        self.block5 = block5
        self.head = head

    def forward(self, x):
        return self.head(self.block5(x))
