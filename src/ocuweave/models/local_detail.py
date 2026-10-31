"""Experimental image-conditioned residual refinement, not a novelty claim."""
import torch
from torch import nn
from torch.nn import functional as F
from ocuweave.models.segmenter import OcuWeaveSegmenter


class LocalDetailSegmenter(OcuWeaveSegmenter):
    """Retain the query network and refine its logits at half input resolution.

    This is a shallow local convolutional branch, not a mirrored encoder-decoder.
    Zero initialization of the final projection preserves the base prediction at
    initialization. The extra parameters/compute must be reported in comparisons.
    """
    def __init__(self, width=24, dim=48, queries=8):
        super().__init__(width, dim, queries)
        self.detail = nn.Sequential(
            nn.Conv2d(3, width, 3, stride=2, padding=1),
            nn.GroupNorm(1, width), nn.GELU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.GroupNorm(1, width), nn.GELU())
        self.refine = nn.Sequential(nn.Conv2d(width + 1, width, 1), nn.GELU(),
                                    nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, images, evidence=None, available=None):
        coarse = super().forward(images, evidence, available)
        local = self.detail(images)
        context = F.interpolate(coarse, size=local.shape[-2:], mode='bilinear', align_corners=False)
        delta = self.refine(torch.cat((local, context), dim=1))
        return coarse + F.interpolate(delta, size=images.shape[-2:], mode='bilinear', align_corners=False)
