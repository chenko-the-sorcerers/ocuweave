"""Smoke results cannot bypass CUDA/provenance gates or access test images."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from ocuweave.data.adapters.otfid import audit_otfid
from ocuweave.training.exploratory import smoke,execute,lock
from ocuweave.data.segmentation import SegmentationDataset
from tests.data.test_segmentation_pipeline import fixture


class ExploratoryTests(unittest.TestCase):
    def test_cuda_absence_fails_before_reading_inputs(self):
        with patch('torch.cuda.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'CUDA required'):smoke('missing','missing','nonexistent-output')

    def test_cpu_smoke_is_not_a_gpu_pass(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=fixture(root/'input.zip',count=64)
            audit_otfid(archive,root/'audit',allow_unverified_patients=True)
            config=dict(architecture='query_detail',width=4,dim=8,queries=2,size=32,batch_size=8,epochs=20,learning_rate=.001,seed=42)
            observed=[];original=SegmentationDataset.__getitem__
            def reader(dataset,index):
                observed.append(dataset.rows[index]['split']);return original(dataset,index)
            with patch('ocuweave.training.exploratory.revision',return_value='synthetic'), \
                 patch('ocuweave.training.exploratory.configurations',return_value={'detail384-seed42':config}), \
                 patch.object(SegmentationDataset,'__getitem__',reader):
                result=smoke(root/'audit/manifest.json',archive,root/'smoke',device='cpu')
            self.assertEqual(set(observed),{'train'})
            self.assertFalse(result['gpu_gate_passed']);self.assertEqual(result['runs'][0]['batches'],3)
            self.assertTrue(result['runs'][0]['checkpoint_reload_matches'])
            self.assertTrue((root/'smoke/detail384-seed42-pairs.png').exists())
            with patch('ocuweave.training.exploratory.lock',return_value={}):
                with self.assertRaisesRegex(ValueError,'CUDA smoke'):execute('missing','missing',root,root/'smoke/smoke.json')
            with self.assertRaises(FileExistsError):smoke('missing','missing',root/'smoke',device='cpu')

    def test_stale_smoke_and_historical_manifest_cannot_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);report=root/'smoke.json'
            report.write_text(json.dumps({'status':'passed','gpu_gate_passed':True,'source_commit':'old'}))
            with patch('ocuweave.training.exploratory.lock',return_value={'source_commit':'new'}):
                with self.assertRaisesRegex(ValueError,'mismatch'):execute('missing','missing',root,report)
            with patch('ocuweave.training.exploratory.checked_manifest',return_value={}):
                with self.assertRaisesRegex(ValueError,'manifest required'):lock('missing','missing',root)
