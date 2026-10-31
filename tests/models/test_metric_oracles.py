"""Hand-computed evaluator oracles, not comparisons against the same formula."""
import unittest
import numpy as np
import torch
from ocuweave.evaluation.segmentation import binary_metrics, aggregate
from ocuweave.evaluation.original_grid import restore_probabilities

class MetricOracleTests(unittest.TestCase):
    def test_partial_overlap_and_false_positive_area(self):
        target = np.array([[1,1,0],[0,0,0]], dtype=bool)
        pred = np.array([[1,0,1],[0,0,0]], dtype=bool)
        metrics = binary_metrics(pred, target)
        self.assertEqual((metrics['tp'],metrics['fp'],metrics['fn']), (1,1,1))
        self.assertEqual(metrics['dice'], .5)
        self.assertAlmostEqual(metrics['iou'], 1/3)
        self.assertAlmostEqual(metrics['fp']/target.size, 1/6)

    def test_empty_and_disjoint_conventions(self):
        zero = np.zeros((2,2), bool)
        one = zero.copy(); one[0,0] = True
        other = zero.copy(); other[1,1] = True
        for pred, target, expected in [(zero,zero,1),(one,zero,0),(zero,one,0),(one,other,0),(one,one,1)]:
            m = binary_metrics(pred,target)
            self.assertEqual(m['dice'], expected); self.assertEqual(m['iou'], expected)

    def test_padding_exclusion_and_positive_macro(self):
        m = binary_metrics([1,1,1,0], [1,0,0,0], [1,0,0,0])
        self.assertEqual(m['fp'], 0); self.assertEqual(m['dice'], 1)
        result = aggregate([binary_metrics([0,0],[1,0]),binary_metrics([0,0],[0,0])])
        self.assertEqual(result['macro_dice'], .5)
        self.assertEqual(result['positive_macro_dice'], 0)
        self.assertEqual(result['empty_image_false_positive_rate'], 0)

    def test_crop_before_bilinear_and_threshold_after_restore(self):
        # Padding is deliberately 1; interior is [0,1]. Aligned-corners=False
        # 2->4 interpolation gives [0,.25,.75,1], independently specified here.
        p = torch.ones(1,1,4,4); p[:,:,1:3,1:3] = torch.tensor([[0.,1.],[0.,1.]])
        valid = np.zeros((4,4),bool); valid[1:3,1:3] = True
        restored = restore_probabilities(p,valid,(2,4))
        expected = torch.tensor([[0.,.25,.75,1.],[0.,.25,.75,1.]])
        torch.testing.assert_close(restored,expected,rtol=0,atol=0)
        np.testing.assert_array_equal(restored.numpy()>=.5, [[0,0,1,1],[0,0,1,1]])
        p[:,:,1:3,1:3] = .5
        self.assertTrue((restore_probabilities(p,valid,(3,7))>=.5).all())

    def test_invalid_inverse_geometry_rejected(self):
        p = torch.zeros(1,1,4,4); valid = np.ones((4,4),bool)
        with self.assertRaises(ValueError): restore_probabilities(p,np.zeros_like(valid),(2,2))
        valid[1,1] = False
        with self.assertRaises(ValueError): restore_probabilities(p,valid,(2,2))
        with self.assertRaises(ValueError): restore_probabilities(p+float('nan'),np.ones_like(valid),(2,2))
