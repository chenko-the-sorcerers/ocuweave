"""New software versions must not disappear from the consolidated evidence report."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    'research_archive', Path(__file__).resolve().parents[1] / 'scripts/archive_research_record.py')
archive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(archive)

class ResearchArchiveTests(unittest.TestCase):
    def test_multiple_versions_and_pending_results_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixtures = {
                '02 — Audits/exploratory-v0.6.0/old/session-status.json': {'completed_runs': 0},
                '02 — Audits/exploratory-v0.8.1/new/session-status.json': {'completed_runs': 0},
                '02 — Audits/exploratory-v0.8.1/new/development-manifest/manifest.json': {'records': [{'sample_id': 'private-example'}]},
                '02 — Audits/cpu-review-v0.7.0/review/summary.json': {'samples': 238},
                '04 — Runs/v0.8.1/new/resolution-detail-seeds/summary.json': {'runs': [{'status': 'pending', 'dice': None}]},
            }
            for relative, value in fixtures.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps(value))
            reports = archive.measured_reports(root)
            lookup = {report['source']: report for report in reports}
            self.assertEqual(set(lookup), set(fixtures))
            self.assertEqual(len(reports), len(fixtures))
            for report in reports:
                self.assertEqual(len(report['sha256']), 64)
            manifest = lookup['02 — Audits/exploratory-v0.8.1/new/development-manifest/manifest.json']
            self.assertEqual(manifest['result']['records']['rows'], 1)
            self.assertNotIn('private-example', json.dumps(manifest))
            suite = lookup['04 — Runs/v0.8.1/new/resolution-detail-seeds/summary.json']
            self.assertIsNone(suite['result']['runs'][0]['dice'])

if __name__ == '__main__':
    unittest.main()
