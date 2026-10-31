import tempfile
import unittest
from pathlib import Path
from ocuweave.experiments.roadmap import make_plan, write_plan

class CrossTaskPlanTests(unittest.TestCase):
    def test_paired_core_changes_only_resolution_or_detail(self):
        plan=make_plan(); rows=plan['runs']
        self.assertEqual(len(rows),45)
        self.assertEqual(len({r['run_id'] for r in rows}),45)
        for track in plan['tracks']:
            core=[r for r in rows if r['track']==track and r['family']=='core']
            self.assertEqual(len(core),9)
            for seed in (42,123,2026):
                group=[r for r in core if r['seed']==seed]
                self.assertEqual([(r['architecture'],r['size']) for r in group],
                                 [('query',192),('query',384),('query_detail',384)])
                self.assertEqual({r['optimizer_steps'] for r in group},{500})
                self.assertEqual({tuple(r['targets']) for r in group},
                                 {tuple(plan['tracks'][track]['targets'])})
    def test_no_invented_results_or_executable_multiclass_claim(self):
        plan=make_plan(); lookup={r['run_id']:r for r in plan['runs']}
        for row in lookup.values():
            self.assertIsNone(row['metric_value'])
            self.assertIsNone(row['checkpoint_sha256'])
            self.assertFalse(row['executable_config'])
            self.assertFalse(row['unet']); self.assertFalse(row['emr'])
            if row['family']=='conditional':
                control=lookup[row['paired_control']]
                for field in ('seed','size','architecture','optimizer_steps','targets'):
                    self.assertEqual(row[field],control[field])
        self.assertEqual(len(plan['evaluation_conditions']),7)
        self.assertEqual(plan['completed_research_runs'],0)
    def test_output_cannot_overwrite_prior_register(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)/'plan'; write_plan(root)
            before=(root/'plan.json').read_bytes()
            with self.assertRaises(FileExistsError): write_plan(root)
            self.assertEqual(before,(root/'plan.json').read_bytes())
