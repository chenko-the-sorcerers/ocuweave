"""Full-coverage tiled inference geometry; no label-dependent crop selection."""

import numpy as np


def tile_boxes(height, width, size, overlap=0):
    if (
        any(type(x) is not int for x in (height, width, size, overlap))
        or min(height, width, size) < 1
        or not 0 <= overlap < size
    ):
        raise ValueError("Positive integer dimensions and 0 <= overlap < size required")

    def starts(length):
        if length <= size:
            return [0]
        values = list(range(0, length - size + 1, size - overlap))
        if values[-1] != length - size:
            values.append(length - size)
        return values

    return [
        (y, x, min(y + size, height), min(x + size, width))
        for y in starts(height)
        for x in starts(width)
    ]


def merge_tiles(predictions, boxes, shape):
    """Average overlapping CxHxW probabilities; reject uncovered original pixels."""
    if (
        not predictions
        or len(predictions) != len(boxes)
        or len(shape) != 2
        or min(shape) < 1
    ):
        raise ValueError("Matching nonempty tiles and positive original shape required")
    channels = np.asarray(predictions[0]).shape[0]
    total = np.zeros((channels, *shape), dtype=np.float64)
    count = np.zeros(shape, dtype=np.uint32)
    for p, (y, x, yy, xx) in zip(predictions, boxes):
        p = np.asarray(p)
        if not (0 <= y < yy <= shape[0] and 0 <= x < xx <= shape[1]) or p.shape != (
            channels,
            yy - y,
            xx - x,
        ):
            raise ValueError("Tile and destination geometry mismatch")
        if not np.isfinite(p).all() or (p < 0).any() or (p > 1).any():
            raise ValueError("Finite probabilities required")
        total[:, y:yy, x:xx] += p
        count[y:yy, x:xx] += 1
    if not count.all():
        raise ValueError("Tile plan leaves uncovered pixels")
    return (total / count[None]).astype(np.float32)
