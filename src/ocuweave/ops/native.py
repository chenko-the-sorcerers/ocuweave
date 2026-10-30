"""Explicit ctypes bridge to the optional C++ CPU library; no silent fallback."""
import ctypes
from pathlib import Path

class NativeOps:
    def __init__(self, library: str | Path):
        self._library = ctypes.CDLL(str(Path(library).resolve()))
        self._mean = self._library.ow_masked_weighted_mean
        self._mean.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                              ctypes.POINTER(ctypes.c_ubyte), ctypes.c_size_t,
                              ctypes.POINTER(ctypes.c_double)]
        self._mean.restype = ctypes.c_int

    def masked_weighted_mean(self, values, weights, available):
        if not (len(values) == len(weights) == len(available)) or not values:
            raise ValueError("inputs must have equal non-zero lengths")
        if any(x not in (False, True, 0, 1) for x in available):
            raise ValueError("availability must be boolean")
        n = len(values)
        output = ctypes.c_double()
        status = self._mean((ctypes.c_double * n)(*values), (ctypes.c_double * n)(*weights),
                            (ctypes.c_ubyte * n)(*available), n, ctypes.byref(output))
        if status:
            raise ValueError("invalid or unavailable evidence in native operator")
        return output.value
