"""Gradient, missing evidence, and real train/evaluate round trip checks."""
import json
from pathlib import Path
import tempfile
import unittest
import torch
from ocuweave.models.segmenter import OcuWeaveSegmenter
from ocuweave.losses.segmentation import segmentation_loss
from ocuweave.training.engine import train, evaluate, checked_manifest
from ocuweave.data.adapters.otfid import audit_otfid
from tests.data.test_segmentation_pipeline import fixture


class SegmenterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_gradient_odd_shape_and_missing_nan_evidence(self):
        torch.manual_seed(10)
        model = OcuWeaveSegmenter(width=4, dim=8, queries=2)
        image = torch.rand(2,3,33,47)
        baseline = model(image)
        missing = model(image, torch.full((2,3,8), float('nan')), torch.zeros(2,3,dtype=torch.bool))
        torch.testing.assert_close(baseline, missing, atol=1e-6, rtol=1e-5)
        loss = segmentation_loss(baseline, torch.zeros_like(baseline))
        loss.backward()
        self.assertEqual(baseline.shape, (2,1,33,47))
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        with self.assertRaises(ValueError):
            model(image, torch.full((2,1,8),float('nan')), torch.ones(2,1,dtype=torch.bool))

    def test_train_evaluate_and_checkpoint_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = fixture(root/'source.zip')
            audit_otfid(archive, root/'audit', allow_unverified_patients=True)
            manifest = root/'audit/manifest.json'
            config = dict(architecture='query',width=4,dim=8,queries=2,size=32,batch_size=8,epochs=1,learning_rate=.001,seed=42)
            for architecture in ('query','fcn'):
                config['architecture'] = architecture
                checkpoint = train(manifest,archive,root/architecture,config,allow_unverified=True,device='cpu')
                result = evaluate(manifest,archive,checkpoint,root/f'{architecture}.json',allow_unverified=True,device='cpu')
                self.assertEqual(result['split'], 'test')
                self.assertGreater(result['metrics']['samples'],0)
                with self.assertRaises(FileExistsError):
                    evaluate(manifest,archive,checkpoint,root/f'{architecture}.json',allow_unverified=True)
            data=json.loads(manifest.read_text());data['records'][1]['sample_id']=data['records'][0]['sample_id']
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError,'Duplicate sample'):
                checked_manifest(manifest,archive,True)

    def test_padding_loss_has_zero_gradient(self):
        logits=torch.zeros(1,1,16,16,requires_grad=True)
        valid=torch.ones_like(logits);valid[:,:,:4]=0
        segmentation_loss(logits,torch.zeros_like(logits),valid).backward()
        self.assertTrue((logits.grad[:,:,:4]==0).all())


if __name__ == '__main__':
    unittest.main()
