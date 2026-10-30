"""Synthetic archive checks: exclusions, grouping, geometry, and metric edge cases."""
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
import numpy as np
from PIL import Image
from ocuweave.data.adapters.otfid import audit_otfid
from ocuweave.data.splitting.groups import assign_splits
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.segmentation import binary_metrics, aggregate


def png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format='PNG')
    return buffer.getvalue()


def fixture(path, count=24):
    rng = np.random.default_rng(7)
    labels = ['image,label']
    with zipfile.ZipFile(path, 'w') as archive:
        for i in range(count):
            name = f'eye_{i:03}'
            healthy = i % 2 == 0
            labels.append(f'{name}.png,{"healthy" if healthy else "active"}')
            archive.writestr(f'images/{name}.png', png(rng.integers(0, 256, (24, 32, 3), dtype=np.uint8)))
            if not healthy:
                mask = np.zeros((24, 32), dtype=np.uint8)
                mask[5:16, 8:22] = 255
                archive.writestr(f'masks/{name}-a.png', png(mask))
        labels.append('missing.png,active')
        archive.writestr('dataset_labels.csv', '\n'.join(labels))
    return path


class DataPipelineTests(unittest.TestCase):
    def test_unknown_patient_guard_and_frozen_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); archive = fixture(root / 'source.zip')
            with self.assertRaisesRegex(ValueError, 'Patient IDs'):
                audit_otfid(archive, root / 'strict')
            report = audit_otfid(archive, root / 'audit', allow_unverified_patients=True)
            self.assertEqual(report['eligible_samples'], 24)
            self.assertEqual(report['exclusions'], [{'sample_id':'missing','reason':'missing_image'}])
            data = json.loads((root / 'audit/manifest.json').read_text())
            self.assertFalse(data['patient_identity_verified'])
            self.assertEqual(set(r['split'] for r in data['records']), {'train','validation','test'})
            with self.assertRaises(FileExistsError):
                audit_otfid(archive, root / 'audit', allow_unverified_patients=True)

    def test_patient_and_duplicate_bridges_cannot_cross_splits(self):
        rows = [dict(sample_id=str(i), pixel_sha256=str(i), label='lesion', patient_id=f'p{i}') for i in range(30)]
        rows[0]['related_patient_ids'] = ['p0','p1']
        rows[1]['similarity_group'] = rows[2]['similarity_group'] = 'similar'
        assign_splits(rows)
        self.assertEqual(len({r['split_group'] for r in rows[:3]}), 1)
        self.assertEqual(len({r['split'] for r in rows[:3]}), 1)

    def test_padding_is_not_background_supervision(self):
        image = Image.new('RGB', (32,16), (100,120,130))
        _, mask, valid = letterbox(image, np.ones((16,32),bool), 32)
        self.assertEqual(valid.sum(), 512)
        self.assertEqual(mask.sum(), 512)
        prediction = np.ones((32,32),bool)
        self.assertEqual(binary_metrics(prediction,mask,valid)['dice'], 1)

    def test_metrics_empty_and_known_overlap(self):
        row = binary_metrics([1,1,0,0], [1,0,1,0])
        self.assertEqual(row['dice'], .5)
        self.assertAlmostEqual(row['iou'], 1/3)
        empty = binary_metrics([0,0], [0,0])
        self.assertEqual(empty['dice'], 1)
        self.assertEqual(aggregate([empty])['positive_macro_dice'], None)
        self.assertEqual(aggregate([empty])['empty_image_false_positive_rate'], 0)


if __name__ == '__main__':
    unittest.main()
