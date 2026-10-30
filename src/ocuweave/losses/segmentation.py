"""Binary BCE + soft Dice. Empty-target images remain supervised by BCE."""
import torch
from torch.nn import functional as F


def segmentation_loss(logits, targets, valid=None):
    if logits.shape != targets.shape:
        raise ValueError('Prediction and target shapes must match')
    valid = torch.ones_like(targets) if valid is None else valid
    if valid.shape != targets.shape or valid.sum() <= 0:
        raise ValueError('A nonempty valid-pixel mask must match targets')
    bce = (F.binary_cross_entropy_with_logits(logits, targets, reduction='none') * valid).sum() / valid.sum()
    p, y = (logits.sigmoid() * valid).flatten(1), (targets * valid).flatten(1)
    dice = (2 * (p * y).sum(1) + 1) / (p.sum(1) + y.sum(1) + 1)
    return bce + (1 - dice).mean()
