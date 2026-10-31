"""Availability-masked evidence tokens for conditioning learned visual queries."""

import torch
from torch import nn


class EvidenceFusion(nn.Module):
    def __init__(self, dim=48, modalities=2):
        super().__init__()
        self.modalities = modalities
        self.types = nn.Parameter(torch.randn(modalities, dim) * 0.02)
        self.norm = nn.LayerNorm(dim)

    def forward(self, representations, available):
        if (
            representations.ndim != 3
            or representations.shape[1:] != self.types.shape
            or available.shape != representations.shape[:2]
            or available.dtype != torch.bool
        ):
            raise ValueError("Expected BxMxD evidence and BxM availability")
        safe = torch.where(
            available[..., None], representations, torch.zeros_like(representations)
        )
        if not torch.isfinite(safe).all():
            raise ValueError("Available evidence must be finite")
        encoded = self.norm(safe + self.types[None])
        return (
            torch.where(available[..., None], encoded, torch.zeros_like(encoded)),
            available,
        )
