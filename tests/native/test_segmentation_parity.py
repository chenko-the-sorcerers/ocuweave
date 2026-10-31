import os
import unittest
import numpy as np
from ocuweave.ops.native import NativeOps
from ocuweave.evaluation.segmentation import binary_metrics


@unittest.skipUnless(
    os.environ.get("OCUWEAVE_NATIVE_LIBRARY"),
    "Build the native library to run native parity",
)
class NativeSegmentationParity(unittest.TestCase):
    def test_random_masks_and_hand_computed_counts(self):
        op = NativeOps(os.environ["OCUWEAVE_NATIVE_LIBRARY"])
        self.assertEqual(
            op.confusion_counts([1, 1, 0, 0, 1], [1, 0, 1, 0, 0], [1, 1, 1, 1, 0]),
            dict(tp=1, fp=1, fn=1, tn=1),
        )
        rng = np.random.default_rng(42)
        for _ in range(20):
            p = rng.random((21, 33)) > 0.6
            y = rng.random((21, 33)) > 0.7
            v = rng.random((21, 33)) > 0.2
            actual = op.binary_metrics(p, y, v)
            reference = binary_metrics(p, y, v)
            for key, value in reference.items():
                self.assertEqual(actual[key], value)
        with self.assertRaises(ValueError):
            op.confusion_counts([2], [0])
        with self.assertRaises(ValueError):
            op.confusion_counts([0], [0], [0])
        empty = op.binary_metrics([0], [0])
        self.assertEqual(empty["dice"], 1)
