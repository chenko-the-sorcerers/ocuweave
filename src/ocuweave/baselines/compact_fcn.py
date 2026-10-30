"""Encoder/pixel-decoder matched ablation: a convolution replaces lesion queries."""
from torch import nn
from torch.nn import functional as F
from ocuweave.models.vision.compact import CompactEncoder
from ocuweave.models.heads.query_mask import PyramidPixels


class CompactFCN(nn.Module):
    def __init__(self, width=24, dim=48, queries=8):
        super().__init__()
        self.encoder = CompactEncoder(width)
        self.pixels = PyramidPixels(self.encoder.channels, dim)
        self.head = nn.Conv2d(dim, 1, 1)
        nn.init.constant_(self.head.bias, -2.0)

    def forward(self, images):
        pixels, _ = self.pixels(self.encoder(images))
        return F.interpolate(self.head(pixels), size=images.shape[-2:],
                             mode='bilinear', align_corners=False)
