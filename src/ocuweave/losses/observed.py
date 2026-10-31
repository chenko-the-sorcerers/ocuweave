"""Auditable BCE/Dice components for observed image-class pairs only."""

import torch
from torch.nn import functional as F


def observed_components(logits, targets, valid, available):
    if logits.shape != targets.shape or logits.ndim != 4:
        raise ValueError("Expected matching BxCxHxW tensors")
    if valid.shape != (logits.shape[0], 1, *logits.shape[-2:]):
        raise ValueError("Valid content mask must be Bx1xHxW")
    if (
        available.shape != logits.shape[:2]
        or available.dtype != torch.bool
        or not available.any()
    ):
        raise ValueError("Explicit nonempty boolean class availability required")
    z = logits[available]
    y = targets[available].float()
    v = valid.expand_as(targets)[available].bool()
    if not v.flatten(1).any(1).all():
        raise ValueError("Every supervised pair needs valid pixels")
    if (
        not torch.isfinite(z[v]).all()
        or not torch.isfinite(y[v]).all()
        or ((y[v] < 0) | (y[v] > 1)).any()
    ):
        raise ValueError("Finite logits and targets in [0, 1] required")
    z = torch.where(v, z, torch.zeros_like(z))
    y = torch.where(v, y, torch.zeros_like(y))
    bce = (
        F.binary_cross_entropy_with_logits(z, y, reduction="none") * v
    ).sum() / v.sum()
    p = (z.sigmoid() * v).flatten(1)
    yf = y.flatten(1)
    dice_loss = (1 - (2 * (p * yf).sum(1) + 1) / (p.sum(1) + yf.sum(1) + 1)).mean()
    return {"total": bce + dice_loss, "bce": bce, "soft_dice": dice_loss}
