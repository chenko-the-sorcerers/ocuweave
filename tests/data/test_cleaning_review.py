import unittest
import numpy as np
from PIL import Image,ImageFilter
from ocuweave.data.auditing.quality_screen import quality_metrics,fit_thresholds,quality_flags,descriptor,near_pairs
from ocuweave.data.auditing.note_context import context_flags,review_context

class CleaningReviewTests(unittest.TestCase):
    def test_blur_proxy_and_train_only_thresholds(self):
        rng=np.random.default_rng(3);image=Image.fromarray(rng.integers(25,220,(256,256),dtype=np.uint8))
        a=quality_metrics(image);b=quality_metrics(image.filter(ImageFilter.GaussianBlur(3)))
        self.assertGreater(a['sharpness'],b['sharpness'])
        thresholds=fit_thresholds([dict(a,split='train'),dict(b,split='validation')])
        self.assertEqual(thresholds['sharpness']['p10'],a['sharpness'])
        self.assertTrue(quality_flags(b,thresholds)['low_sharpness_proxy'])

    def test_transformed_duplicates_and_pair_cap(self):
        rng=np.random.default_rng(9);a=rng.integers(20,200,(128,128),dtype=np.uint8)
        rows=[{'sample_id':str(i),'split':'train' if i==0 else 'test'} for i in range(3)]
        values=[descriptor(Image.fromarray(x)) for x in (a,a[:,::-1],rng.integers(20,200,(128,128),dtype=np.uint8))]
        result=near_pairs(rows,values)
        self.assertEqual(len(result['pairs']),1);self.assertTrue(result['pairs'][0]['cross_split'])
        self.assertEqual(result['pairs'][0]['transform_of_b'],'horizontal_flip')
        capped=near_pairs(rows,[values[0]]*3,limit=1)
        self.assertTrue(capped['truncated'])
        self.assertIsNone(descriptor(Image.new('L',(64,64),100)))

    def test_clause_context_and_no_raw_note_export(self):
        note='No glaucoma. Mother has glaucoma; possible glaucoma OD. Prior glaucoma treatment. But definite glaucoma.'
        flags=context_flags(note)
        self.assertEqual(len(flags['mentions']),5)
        self.assertTrue(flags['mentions'][0]['negation_near_mention'])
        self.assertTrue(flags['mentions'][1]['family_near_mention'])
        self.assertTrue(flags['mentions'][2]['suspected_near_mention'])
        self.assertTrue(flags['mentions'][3]['historical_near_mention'])
        self.assertFalse(flags['mentions'][4]['negation_near_mention'])
        for m in flags['mentions']:self.assertEqual(note[m['start']:m['end']].lower(),'glaucoma')
        class Secret(dict):
            def __getitem__(self,key):
                if key=='note':raise AssertionError('Test note read')
                return super().__getitem__(key)
        result=review_context([{'filename':'x','use':'training','note':note},Secret(use='test')])
        self.assertNotIn(note,str(result));self.assertFalse(result['training_cleared'])
        self.assertEqual(result['counts']['training']['rows'],1)

    def test_fairvlmed_end_to_end_preserves_sources_and_excludes_test_review(self):
        import csv, io, json, tempfile, zipfile
        from pathlib import Path
        from scripts.review_cleaning import fairvlmed
        from ocuweave.utils.provenance import sha256
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'Dataset').mkdir();(root/'ReadMe').mkdir()
            table=root/'ReadMe/data_summary.csv';archive=root/'Dataset/dataset.zip'
            with table.open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=['filename','use','note','glaucoma'])
                writer.writeheader()
                for i,split in enumerate(('training','validation','test')):
                    writer.writerow(dict(filename=f'{i}.npz',use=split,note='No glaucoma' if i<2 else 'PRIVATE TEST SENTINEL',glaucoma='no'))
            rng=np.random.default_rng(12);pixels=rng.integers(20,200,(128,128),dtype=np.uint8)
            with zipfile.ZipFile(archive,'w') as source:
                for i in range(3):
                    stream=io.BytesIO();np.savez_compressed(stream,slo_fundus=pixels if i<2 else pixels[:,::-1])
                    source.writestr(f'{i}.npz',stream.getvalue())
            hashes={name:sha256(root/name) for name in ('Dataset/dataset.zip','ReadMe/data_summary.csv')}
            (root/'download_manifest.json').write_text(json.dumps({'files':[{'path':k,'sha256':v} for k,v in hashes.items()]}))
            report=fairvlmed(root,root/'review')
            self.assertEqual(report['quality']['samples'],2)
            self.assertEqual(report['fingerprinted_images'],3)
            self.assertEqual(report['cross_split_candidates'],3)
            self.assertFalse(report['duplicate_scan_truncated'])
            queue=(root/'review/emr-clinical-review.csv').read_text()
            self.assertNotIn('2.npz',queue)
            for path in (root/'review').glob('*.json'):
                self.assertNotIn('PRIVATE TEST SENTINEL',path.read_text())
            self.assertEqual(hashes,{name:sha256(root/name) for name in hashes})

    def test_archive_includes_nested_cleaning_evidence_without_pair_rows(self):
        import json,tempfile
        from pathlib import Path
        from scripts.archive_research_record import measured_reports
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);audit=root/'02 — Audits/cleaning-review-v0.5.0/run/review'
            (audit/'otfid').mkdir(parents=True)
            (audit/'summary.json').write_text(json.dumps({'automatic_exclusions':0}))
            (audit/'otfid/near-duplicates.json').write_text(json.dumps({'pairs':[{'a':'private-id'}],'truncated':False}))
            reports=measured_reports(root)
            self.assertEqual(len(reports),2)
            self.assertNotIn('private-id',json.dumps(reports))
            pair=next(r for r in reports if r['source'].endswith('near-duplicates.json'))
            self.assertEqual(pair['result']['pairs']['rows'],1)
            self.assertFalse(pair['result']['truncated'])
