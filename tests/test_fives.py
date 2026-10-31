import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from ocuweave.data.adapters.fives import prepare_fives, FivesDataset, verify_fives
from ocuweave.training.vessels import vessel_metrics


class FivesChecks(unittest.TestCase):
    def fixture(self, root):
        rng=np.random.default_rng(12)
        for split,n in [('train',12),('test',2)]:
            for folder in ['Original','Ground truth']:(root/split/folder).mkdir(parents=True)
            for i in range(n):
                image=rng.integers(0,255,(32,48,3),dtype=np.uint8)
                mask=np.zeros((32,48),dtype=np.uint8);mask[:,20:23]=255
                Image.fromarray(image).save(root/split/'Original'/f'{i}_N.png')
                Image.fromarray(mask).save(root/split/'Ground truth'/f'{i}_N.png')

    def test_partition_and_tamper_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); self.fixture(root); path=root/'manifest.json'
            m=prepare_fives(root,path);self.assertEqual(m['counts'],{'test':2,'train':10,'validation':2})
            self.assertTrue(all(r['split']=='test' for r in m['records'] if r['official_split']=='test'))
            verify_fives(root,m)
            sample=FivesDataset(root,m,'train',32)[0]
            self.assertEqual(tuple(sample['image'].shape),(3,32,32))
            self.assertFalse((sample['mask'].bool() & ~sample['valid'].bool()).any())
            with self.assertRaises(FileExistsError):prepare_fives(root,path)
            (root/m['records'][0]['mask']).write_bytes(b'changed')
            with self.assertRaises(ValueError):verify_fives(root,m)

    def test_duplicate_cross_partition_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            (root/'test/Original/0_N.png').write_bytes((root/'train/Original/0_N.png').read_bytes())
            with self.assertRaises(ValueError):prepare_fives(root,root/'manifest.json')

    def test_wrong_mask_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);self.fixture(root)
            Image.new('L',(16,16)).save(root/'train/Ground truth/0_N.png')
            with self.assertRaises(ValueError):prepare_fives(root,root/'manifest.json')

    def test_vessel_metric_oracles(self):
        target=np.zeros((16,16),dtype=bool);target[2:14,8]=True
        perfect=vessel_metrics(target,target)
        for key in ('dice','iou','cldice','thin_centerline_recall'):self.assertEqual(perfect[key],1)
        empty=vessel_metrics(np.zeros_like(target),target)
        for key in ('dice','iou','cldice','thin_centerline_recall'):self.assertEqual(empty[key],0)
        self.assertIsNone(empty['precision'])
        pred=target.copy();pred[0,0]=True
        self.assertEqual(vessel_metrics(pred,target)['false_positive_area_fraction'],1/256)
        both=vessel_metrics(np.zeros_like(target),np.zeros_like(target))
        self.assertEqual(both['cldice'],1);self.assertIsNone(both['thin_centerline_recall'])


if __name__=='__main__':unittest.main()
