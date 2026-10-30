"""OcuWeave compact query segmenter. Research prototype, not a validated diagnostic model."""
from torch import nn
from .vision.compact import CompactEncoder
from .heads.query_mask import QueryMaskHead


class OcuWeaveSegmenter(nn.Module):
    def __init__(self, width=24, dim=48, queries=8):
        super().__init__()
        if dim % 4 or width < 4 or queries < 1:
            raise ValueError('dim must be divisible by four; width >= 4; queries >= 1')
        self.encoder = CompactEncoder(width)
        self.head = QueryMaskHead(self.encoder.channels, dim, queries)

    def forward(self, images, evidence=None, available=None):
        if images.ndim != 4 or images.shape[1] != 3 or min(images.shape[-2:]) < 16:
            raise ValueError('Expected Bx3xHxW images, H and W >= 16')
        return self.head(self.encoder(images), images.shape[-2:], evidence, available)
