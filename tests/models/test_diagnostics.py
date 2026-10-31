import unittest
from ocuweave.evaluation.diagnostics import area_bin, diagnose


class DiagnosticTests(unittest.TestCase):
    def test_test_split_rejected_before_any_data_or_checkpoint_read(self):
        with self.assertRaisesRegex(ValueError, "test remains sealed"):
            diagnose("missing", "missing", "missing", "missing", split="test")

    def test_area_boundaries_are_explicit(self):
        self.assertEqual([area_bin(x) for x in (0, .001, .01, .05, .051)],
                         ["empty", "small_le_1pct", "small_le_1pct", "medium_1_to_5pct", "large_gt_5pct"])
