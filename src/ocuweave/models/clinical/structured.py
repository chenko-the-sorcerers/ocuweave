"""Training-only normalization and explicitly missing structured clinical features."""

import torch
from torch import nn


class ClinicalEncoder(nn.Module):
    def __init__(self, features, dim=48):
        super().__init__()
        if features < 1 or dim < 1:
            raise ValueError("Positive feature and embedding dimensions required")
        self.features = features
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))
        self.register_buffer("observed_count", torch.zeros(features, dtype=torch.long))
        self.network = nn.Sequential(
            nn.Linear(features * 2, dim), nn.GELU(), nn.Linear(dim, dim)
        )

    def _check(self, values, observed):
        if (
            values.ndim != 2
            or values.shape[1] != self.features
            or values.shape != observed.shape
            or observed.dtype != torch.bool
        ):
            raise ValueError(
                "Expected BxF values with explicit boolean observation mask"
            )
        safe = torch.where(observed, values, torch.zeros_like(values))
        if not torch.isfinite(safe).all():
            raise ValueError("Observed features must be finite")
        return safe

    @torch.no_grad()
    def fit_statistics(self, values, observed, *, splits):
        if (
            len(splits) != len(values)
            or not splits
            or any(s != "train" for s in splits)
        ):
            raise ValueError("Fit statistics on training rows only")
        safe = self._check(values, observed)
        counts = observed.sum(0)
        mean = safe.sum(0) / counts.clamp_min(1)
        delta = torch.where(observed, safe - mean, torch.zeros_like(safe))
        scale = (delta.square().sum(0) / counts.clamp_min(1)).sqrt().clamp_min(1e-6)
        scale = torch.where(counts > 1, scale, torch.ones_like(scale))
        self.mean.copy_(mean)
        self.scale.copy_(scale)
        self.observed_count.copy_(counts)

    def forward(self, values, observed):
        safe = self._check(values, observed)
        normalized = torch.where(
            observed, (safe - self.mean) / self.scale, torch.zeros_like(safe)
        )
        encoded = self.network(
            torch.cat((normalized, observed.to(normalized.dtype)), dim=1)
        )
        available = observed.any(1)
        return (
            torch.where(available[:, None], encoded, torch.zeros_like(encoded)),
            available,
        )
