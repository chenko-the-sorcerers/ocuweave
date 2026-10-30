"""Lesion queries decode a single high-resolution mask feature grid."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class PyramidPixels(nn.Module):
    """One shared-resolution pixel grid, reused by the query model and FCN ablation."""
    def __init__(self, channels, dim):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(c, dim, 1) for c in channels)
        self.mix = nn.Sequential(nn.Conv2d(dim * len(channels), dim, 1),
                                 nn.GroupNorm(1, dim), nn.GELU())

    def forward(self, features):
        projected = [layer(x) for layer, x in zip(self.projections, features)]
        pixels = self.mix(torch.cat([F.interpolate(x, size=projected[0].shape[-2:],
            mode='bilinear', align_corners=False) for x in projected], dim=1))
        return pixels, projected[-1].flatten(2).transpose(1, 2)


class QueryMaskHead(nn.Module):
    def __init__(self, channels, dim=48, queries=8):
        super().__init__()
        self.pixels = PyramidPixels(channels, dim)
        self.query = nn.Parameter(torch.randn(queries, dim) / math.sqrt(dim))
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.mask_embedding = nn.Linear(dim, dim)
        self.query_weight = nn.Linear(dim, 1)
        self.bias = nn.Parameter(torch.tensor(-2.0))

    def forward(self, features, output_size, evidence=None, available=None):
        pixels, context = self.pixels(features)
        q = self.query.unsqueeze(0).expand(context.shape[0], -1, -1)
        padding = None
        if evidence is not None:
            if evidence.ndim != 3 or evidence.shape[0] != context.shape[0] or evidence.shape[-1] != context.shape[-1]:
                raise ValueError('Evidence must be BxTxD with matching batch and embedding size')
            if available is None or available.dtype != torch.bool or available.shape != evidence.shape[:2]:
                raise ValueError('Evidence requires explicit per-token availability')
            # Missing values are ignored before entering attention, including NaN payloads.
            safe = torch.where(available[..., None], evidence, torch.zeros_like(evidence))
            if not torch.isfinite(safe).all():
                raise ValueError('Available evidence must contain finite values')
            context = torch.cat((context, safe), dim=1)
            padding = torch.cat((torch.zeros((context.shape[0], context.shape[1] - safe.shape[1]),
                dtype=torch.bool, device=context.device), ~available), dim=1)
        attended, _ = self.attention(q, context, context, key_padding_mask=padding, need_weights=False)
        q = self.norm(q + attended)
        masks = torch.einsum('bqd,bdhw->bqhw', self.mask_embedding(q), pixels) / math.sqrt(pixels.shape[1])
        weights = self.query_weight(q).softmax(dim=1)
        logits = (masks * weights[..., None]).sum(dim=1, keepdim=True) + self.bias
        return F.interpolate(logits, size=output_size, mode='bilinear', align_corners=False)
