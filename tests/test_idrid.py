import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ocuweave.data.adapters.idrid import CLASSES,prepare_idrid,IdridDataset,read_idrid
from ocuweave.models.multilabel import MultiLabelLQNet
from ocuweave.training.lesions import observed_loss


class IdridChecks(unittest.TestCase):
    def test_unknown_annotation_has_no_gradient(self):
        x=torch.zeros(2,4,16,16,requires_grad=True)
        y=torch.zeros_like(x);valid=torch.ones(2,1,16,16);available=torch.ones(2,4,dtype=torch.bool)
        available[0,2]=False;y[0,2]=float('nan')
        loss=observed_loss(x,y,valid,available);self.assertTrue(torch.isfinite(loss));loss.backward()
        self.assertEqual(float(x.grad[0,2].abs().sum()),0)
        self.assertGreater(float(x.grad[0,1].abs().sum()),0)
        with self.assertRaises(ValueError):observed_loss(x,y,valid,torch.zeros_like(available))

    def test_four_distinct_channels_and_zero_refinement(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        model=MultiLabelLQNet(width=8,dim=16,queries=2)
        x=torch.rand(2,3,32,48);actual=model(x)
        self.assertEqual(tuple(actual.shape),(2,4,32,48))
        features=model.encoder(x)
        coarse=torch.cat([h(features,x.shape[-2:]) for h in model.heads],dim=1)
        torch.testing.assert_close(actual,coarse)
        actual.mean().backward();self.assertTrue(torch.isfinite(model.refine[-1].weight.grad).all())

    def test_palette_indices_are_labels_not_display_luminance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);Image.new('RGB',(8,6)).save(root/'photo.jpg')
            labels=np.zeros((6,8),dtype=np.uint8);labels[2:4,3:6]=1
            mask=Image.fromarray(labels).convert('P')
            mask.putpalette([0,0,0,255,0,0]+[0]*762);mask.save(root/'mask.tif')
            row={'image':'photo.jpg','masks':{c:'mask.tif' if c=='MA' else None for c in CLASSES}}
            _,targets,available=read_idrid(root,row)
            np.testing.assert_array_equal(targets[0],labels>0)
            self.assertEqual(int(targets[0].sum()),6);self.assertEqual(available.tolist(),[True,False,False,False])
            Image.new('RGB',(8,6),'red').save(root/'mask.tif')
            with self.assertRaisesRegex(ValueError,'indexed or scalar'):read_idrid(root,row)

    def test_original_partitions_and_missing_masks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);rng=np.random.default_rng(1)
            for i in range(1,13):
                split='a. Training Set' if i<=10 else 'b. Testing Set'
                folder=root/'A. Segmentation/1. Original Images'/split;folder.mkdir(parents=True,exist_ok=True)
                Image.fromarray(rng.integers(0,255,(32,48,3),dtype=np.uint8)).save(folder/f'IDRiD_{i:02}.jpg')
                for c in CLASSES:
                    if c=='SE':continue
                    maskfolder=root/'A. Segmentation/2. All Segmentation Groundtruths'/split/c;maskfolder.mkdir(parents=True,exist_ok=True)
                    values=np.zeros((32,48),dtype=np.uint8);values[4:8,5:9]=255
                    Image.fromarray(values).save(maskfolder/f'IDRiD_{i:02}_{c}.tif')
            data=prepare_idrid(root,root/'manifest.json')
            self.assertEqual(data['counts'],{'train':8,'validation':2,'test':2})
            self.assertTrue(all(r['masks']['SE'] is None for r in data['records']))
            sample=IdridDataset(root,data,'train',32)[0]
            self.assertEqual(tuple(sample['mask'].shape),(4,32,32));self.assertFalse(sample['available'][3])
            self.assertFalse((sample['mask'].bool() & ~sample['valid'].bool()).any())


if __name__=='__main__':unittest.main()
