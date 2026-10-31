"""Causal ablation controls, original-grid evaluation and geometric review checks."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
from ocuweave.models.segmenter import OcuWeaveSegmenter
from ocuweave.models.local_detail import LocalDetailSegmenter
from ocuweave.losses.segmentation import segmentation_loss
from ocuweave.data.auditing.annotation_geometry import review_annotations, restore_mask
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.data.adapters.otfid import audit_otfid
from ocuweave.evaluation.original_grid import evaluate_original
from ocuweave.training.engine import train
from ocuweave.training.ablation import configurations, summarize
from tests.data.test_segmentation_pipeline import fixture
from ocuweave.utils.provenance import sha256, write_json


class DevelopmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def test_local_refinement_starts_as_base_and_learns(self):
        torch.manual_seed(42);base=OcuWeaveSegmenter(4,8,2)
        torch.manual_seed(42);detail=LocalDetailSegmenter(4,8,2)
        x=torch.rand(2,3,33,47)
        torch.testing.assert_close(base(x),detail(x),atol=0,rtol=0)
        optimizer=torch.optim.AdamW(detail.parameters(),lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            y=detail(x,torch.full((2,1,8),float('nan')),torch.zeros(2,1,dtype=torch.bool))
            segmentation_loss(y,torch.ones_like(y)).backward();optimizer.step()
        self.assertGreater(float(detail.detail[0].weight.grad.abs().sum()),0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in detail.parameters()))

    def test_mask_roundtrip_preserves_native_geometry_and_padding(self):
        mask=np.zeros((16,32),bool);mask[3:8,5:10]=True
        _,resized,valid=letterbox(Image.new('RGB',(32,16)),mask,32)
        np.testing.assert_array_equal(restore_mask(resized,valid,(32,16)),mask)

    def test_annotation_review_and_original_evaluation_do_not_read_test_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);archive=fixture(root/'source.zip')
            audit_otfid(archive,root/'audit',allow_unverified_patients=True)
            manifest=root/'audit/manifest.json';data=json.loads(manifest.read_text())
            selected=next(r['sample_id'] for r in data['records'] if r['split']=='validation')
            from ocuweave.data.auditing.annotation_geometry import source_pair
            with patch('ocuweave.data.auditing.annotation_geometry.source_pair',wraps=source_pair) as reader:
                result=review_annotations(manifest,archive,root/'review',sizes=(32,64),selected=[selected],allow_unverified=True)
                self.assertTrue(all(c.args[1]['split']!='test' for c in reader.call_args_list))
                self.assertEqual(result['pair_failures'],[])
                self.assertEqual(result['by_size']['32']['mean_positive_roundtrip_dice'],1)
                self.assertTrue((root/'review/overlay-00.png').exists())
            test_id=next(r['sample_id'] for r in data['records'] if r['split']=='test')
            with self.assertRaises(ValueError):review_annotations(manifest,archive,root/'bad',selected=[test_id],allow_unverified=True)
            cfg=dict(architecture='query_detail',width=4,dim=8,queries=2,size=32,batch_size=8,epochs=1,learning_rate=.001,seed=42)
            checkpoint=train(manifest,archive,root/'run',cfg,allow_unverified=True,device='cpu')
            with patch('ocuweave.evaluation.original_grid.source_pair',wraps=source_pair) as reader:
                report=evaluate_original(manifest,archive,checkpoint,root/'native.json',allow_unverified=True)
                self.assertTrue(all(c.args[1]['split']=='validation' for c in reader.call_args_list))
            self.assertEqual(report['metrics']['samples'],sum(r['split']=='validation' for r in data['records']))
            timing=json.loads((root/'run/checkpoint.json').read_text())
            self.assertEqual(timing['optimizer_steps_completed'],(sum(r['split']=='train' for r in data['records'])+7)//8)
            self.assertIsNone(timing['peak_cuda_allocated_bytes'])
            write_json(root/'run/validation-original.json',report)
            files=('best.pt','checkpoint.json','provenance.json','history.json','validation-original.json')
            write_json(root/'run/complete.json',{'artifacts':{f:sha256(root/'run'/f) for f in files}})
            registered=summarize(root,{'run':cfg})
            self.assertEqual(registered['runs'][0]['status'],'completed')
            (root/'run/history.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'artifact changed'):summarize(root,{'run':cfg})
            with self.assertRaises(FileExistsError):evaluate_original(manifest,archive,checkpoint,root/'native.json',allow_unverified=True)
        with self.assertRaises(ValueError):evaluate_original('missing','missing','missing','missing',split='test')

    def test_protocol_pairs_and_pending_results(self):
        configs=configurations();self.assertEqual(len(configs),9)
        for seed in (42,123,2026):
            a,b,c=(configs[f'{name}-seed{seed}'] for name in ('query192','query384','detail384'))
            self.assertEqual({k:v for k,v in a.items() if k!='size'},{k:v for k,v in b.items() if k!='size'})
            self.assertEqual({k:v for k,v in b.items() if k!='architecture'},{k:v for k,v in c.items() if k!='architecture'})
        with tempfile.TemporaryDirectory() as directory:
            report=summarize(directory,configs)
            self.assertTrue(all(r['status']=='pending' and 'positive_dice' not in r for r in report['runs']))
            self.assertTrue(all(r['mean_delta'] is None for r in report['paired_comparisons']))


if __name__=='__main__':unittest.main()
