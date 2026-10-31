"""Review integrity, topology, known resize losses and development-only access."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
from ocuweave.data.auditing.review_bundle import component_labels, resize_findings, build_review
from ocuweave.data.auditing.annotation_geometry import source_pair
from ocuweave.data.adapters.otfid import audit_otfid
from ocuweave.utils.provenance import sha256
from tests.data.test_segmentation_pipeline import fixture

class ReviewBundleTests(unittest.TestCase):
    def test_four_connected_diagonals_and_bridging_runs(self):
        self.assertEqual(int(component_labels(np.eye(3,dtype=bool)).max()),3)
        mask = np.array([[1,0,1],[1,1,1],[0,0,0]],bool)
        self.assertEqual(int(component_labels(mask).max()),1)
        self.assertEqual(int(component_labels(np.zeros((3,4),bool)).max()),0)

    def test_single_component_lost_while_whole_mask_survives(self):
        mask = np.zeros((32,32),bool); mask[0,0]=True; mask[16:24,16:24]=True
        result = resize_findings(Image.new('RGB',(32,32)),mask,16)
        self.assertFalse(result['whole_mask_vanished'])
        self.assertEqual(result['original_components_4connected'],2)
        self.assertEqual(result['components_lost_by_sampling'],1)
        self.assertEqual(result['resized_lesion_pixels'],16)

    def test_padding_and_identity_roundtrip(self):
        mask = np.zeros((16,32),bool); mask[4:8,4:8]=True
        result = resize_findings(Image.new('RGB',(32,16)),mask,32)
        self.assertEqual(result['padding_fraction'],.5)
        self.assertEqual(result['mask_pixels_in_padding'],0)
        self.assertEqual(result['roundtrip_dice'],1)
        self.assertEqual(result['exact_boundary_dice'],1)
        self.assertEqual(result['components_lost_by_sampling'],0)

    def test_bundle_keeps_every_development_row_without_reading_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = fixture(root/'source.zip')
            audit_otfid(archive,root/'audit',allow_unverified_patients=True)
            manifest = root/'audit/manifest.json'; before=(sha256(manifest),sha256(archive))
            with patch('ocuweave.data.auditing.review_bundle.source_pair',wraps=source_pair) as reader:
                result = build_review(manifest,archive,root/'review',allow_unverified=True,panel_count=1)
                self.assertTrue(all(c.args[1]['split'] in {'train','validation'} for c in reader.call_args_list))
            self.assertEqual(before,(sha256(manifest),sha256(archive)))
            self.assertEqual(result['automatic_exclusions'],0)
            self.assertFalse(result['clinical_review_complete'])
            self.assertTrue((root/'review/overlay-000.png').exists())
            with (root/'review/image-mask-review.csv').open() as stream:
                rows=list(csv.DictReader(stream))
            self.assertEqual(len(rows),result['development_samples'])
            self.assertTrue(all(r['disposition']=='retain_pending_review' for r in rows))
            with self.assertRaises(FileExistsError): build_review(manifest,archive,root/'review',allow_unverified=True)
