"""Check reference/native parity and report CPU call overhead, not model performance."""
import argparse
import math
import timeit
from ocuweave.ops.native import NativeOps
from ocuweave.ops.reference import masked_weighted_mean

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    native = NativeOps(args.library)
    inputs = ([1., 4., 9.], [1., 2., 3.], [1, 0, 1])
    expected = masked_weighted_mean(*inputs)
    actual = native.masked_weighted_mean(*inputs)
    if not math.isclose(expected, actual, rel_tol=1e-12):
        raise RuntimeError("native/reference mismatch")
    print("CPU example parity passed. Includes Python/ctypes conversion overhead.")
    for name, call in (("reference", lambda: masked_weighted_mean(*inputs)),
                       ("native", lambda: native.masked_weighted_mean(*inputs))):
        print(f"{name}: {timeit.timeit(call, number=10000) / 10000:.8f} seconds/call")

if __name__ == "__main__": main()
