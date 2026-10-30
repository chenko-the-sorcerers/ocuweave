"""Compact convolutional-attention encoder, independently implemented in PyTorch."""
import torch
from torch import nn


class SpatialMixer(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.spatial = nn.Conv2d(channels, channels, 7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.channel = nn.Sequential(nn.Conv2d(channels, channels * 2, 1), nn.GELU(),
                                     nn.Conv2d(channels * 2, channels, 1))
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))

    def forward(self, x):
        return x + self.scale * self.channel(self.norm(self.spatial(x)))


class CompactEncoder(nn.Module):
    """Stride 4/8/16 feature pyramid; no mirrored decoder or U-Net skip path."""
    def __init__(self, width: int = 24):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, width, 4, stride=4),
                                  nn.GroupNorm(1, width), nn.GELU())
        self.stages = nn.ModuleList([SpatialMixer(width),
            nn.Sequential(nn.Conv2d(width, width * 2, 2, stride=2), SpatialMixer(width * 2)),
            nn.Sequential(nn.Conv2d(width * 2, width * 4, 2, stride=2), SpatialMixer(width * 4))])
        self.channels = (width, width * 2, width * 4)

    def forward(self, image):
        x = self.stem(image)
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        return features
