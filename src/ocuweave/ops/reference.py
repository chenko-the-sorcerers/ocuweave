"""CPU interoperability reference, not a differentiable fusion algorithm."""
import math

def masked_weighted_mean(values, weights, available):
    if not (len(values) == len(weights) == len(available)) or not values:
        raise ValueError("inputs must have equal non-zero lengths")
    active = []
    for value, weight, present in zip(values, weights, available):
        if present not in (False, True, 0, 1):
            raise ValueError("availability must be boolean")
        if present:
            if not math.isfinite(value) or not math.isfinite(weight) or weight < 0:
                raise ValueError("available values must be finite and weights non-negative")
            active.append((value, weight))
    scale = max((weight for _, weight in active), default=0)
    if scale <= 0:
        raise ValueError("no positive-weight evidence is available")
    denominator = math.fsum(weight / scale for _, weight in active)
    result = math.fsum(value * ((weight / scale) / denominator) for value, weight in active)
    if not math.isfinite(result):
        raise ValueError("non-finite result")
    return result
