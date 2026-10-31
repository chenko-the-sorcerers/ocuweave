"""Explicit ctypes bridge to the optional C++ CPU library; no silent fallback."""

import ctypes
from pathlib import Path


class NativeOps:
    def __init__(self, library: str | Path):
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        self._mean = self._library.ow_masked_weighted_mean
        self._mean.argtypes = [
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_double),
        ]
        self._mean.restype = ctypes.c_int

    def masked_weighted_mean(self, values, weights, available):
        if not (len(values) == len(weights) == len(available)) or not values:
            raise ValueError("inputs must have equal non-zero lengths")
        if any(x not in (False, True, 0, 1) for x in available):
            raise ValueError("availability must be boolean")
        n = len(values)
        output = ctypes.c_double()
        status = self._mean(
            (ctypes.c_double * n)(*values),
            (ctypes.c_double * n)(*weights),
            (ctypes.c_ubyte * n)(*available),
            n,
            ctypes.byref(output),
        )
        if status:
            raise ValueError("invalid or unavailable evidence in native operator")
        return output.value

    def confusion_counts(self, prediction, target, valid=None):
        """C++17 binary confusion counts; padding is excluded explicitly."""
        import numpy as np

        pred, truth = np.asarray(prediction), np.asarray(target)
        mask = (
            np.ones_like(pred, dtype=np.uint8) if valid is None else np.asarray(valid)
        )
        if pred.shape != truth.shape or mask.shape != pred.shape or not pred.size:
            raise ValueError("Matching nonempty masks required")
        if any(not ((a == 0) | (a == 1)).all() for a in (pred, truth, mask)):
            raise ValueError("Masks must contain only 0/1 values")
        arrays = [np.ascontiguousarray(a, dtype=np.uint8) for a in (pred, truth, mask)]
        pointer = ctypes.POINTER(ctypes.c_ubyte)
        function = self._library.ow_confusion_counts
        function.argtypes = [
            pointer,
            pointer,
            pointer,
            ctypes.c_size_t,
            ctypes.POINTER(ctypes.c_uint64),
        ]
        function.restype = ctypes.c_int
        output = (ctypes.c_uint64 * 4)()
        status = function(
            *(a.ctypes.data_as(pointer) for a in arrays), pred.size, output
        )
        if status:
            raise ValueError("Native metric requires at least one valid pixel")
        return dict(zip(("tp", "fp", "fn", "tn"), output))

    def binary_metrics(self, prediction, target, valid=None):
        counts = self.confusion_counts(prediction, target, valid)
        tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
        return {
            **counts,
            "dice": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 1.0,
            "iou": tp / (tp + fp + fn) if tp + fp + fn else 1.0,
            "target_positive": bool(tp + fn),
            "prediction_positive": bool(tp + fp),
        }
