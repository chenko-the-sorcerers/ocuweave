"""Deterministic paired batch augmentation, independent of model RNG consumption."""

import torch


def horizontal_flip(batch, *, seed, probability=0.5):
    if not 0 <= probability <= 1:
        raise ValueError("Probability must be within [0, 1]")
    result = dict(batch)
    b = batch["image"].shape[0]
    flips = torch.rand(b, generator=torch.Generator().manual_seed(seed)) < probability
    for key in ("image", "mask", "valid"):
        value = batch[key].clone()
        if value.shape[0] != b:
            raise ValueError("Paired tensors require matching batch sizes")
        selected = flips.to(value.device)
        value[selected] = value[selected].flip(-1)
        result[key] = value
    return result
