import math
import os
import random
import unittest
from ocuweave.ops.reference import masked_weighted_mean
from ocuweave.ops.native import NativeOps

class ReferenceTests(unittest.TestCase):
    def test_missing_nan_does_not_contaminate(self):
        self.assertEqual(masked_weighted_mean([2, 8, math.nan], [1, 3, 1], [1, 1, 0]), 6.5)
    def test_no_evidence_is_not_a_prediction(self):
        with self.assertRaises(ValueError): masked_weighted_mean([2], [1], [0])
    def test_negative_weight_rejected(self):
        with self.assertRaises(ValueError): masked_weighted_mean([2], [-1], [1])

@unittest.skipUnless(os.environ.get("OCUWEAVE_NATIVE_LIBRARY"), "set OCUWEAVE_NATIVE_LIBRARY to test native parity")
class NativeParityTests(unittest.TestCase):
    def test_seeded_parity(self):
        native = NativeOps(os.environ["OCUWEAVE_NATIVE_LIBRARY"])
        rng = random.Random(42)
        for _ in range(100):
            values = [rng.uniform(-10, 10) for _ in range(30)]
            weights = [rng.random() for _ in values]
            available = [1] + [rng.randrange(2) for _ in values[1:]]
            self.assertAlmostEqual(native.masked_weighted_mean(values, weights, available),
                                   masked_weighted_mean(values, weights, available), places=12)
    def test_native_rejects_no_evidence(self):
        native = NativeOps(os.environ["OCUWEAVE_NATIVE_LIBRARY"])
        with self.assertRaises(ValueError): native.masked_weighted_mean([2], [1], [0])
