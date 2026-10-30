import csv
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from ocuweave.data.adapters.fairvlmed import audit_fairvlmed


class FairVLMedTests(unittest.TestCase):
    def test_pairing_and_aggregate_only_note_audit(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'ReadMe').mkdir();(root/'Dataset').mkdir()
            with (root/'ReadMe/data_summary.csv').open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['filename','use','note','glaucoma'])
                writer.writeheader()
                for i,split in enumerate(['training','validation','test']):
                    writer.writerow(dict(filename=f'{i}.npz',use=split,note='private glaucoma text',glaucoma='yes'))
            with zipfile.ZipFile(root/'Dataset/dataset.zip','w') as archive:
                for i in range(2):archive.writestr(f'dataset/{i}.npz',b'not decoded by structural audit')
            report=audit_fairvlmed(root,root/'audit')
            self.assertEqual(report['paired_filenames'],2)
            self.assertEqual(report['unmatched_rows'],1)
            self.assertEqual(report['exact_nonempty_notes_across_official_splits'],1)
            self.assertNotIn('private glaucoma text',json.dumps(report))
            self.assertFalse(report['image_decoding_performed'])
