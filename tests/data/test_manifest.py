import json
import tempfile
import unittest
from pathlib import Path
from ocuweave.data.auditing.manifest import audit_manifest
from ocuweave.data.schema import ClinicalSample

class ManifestTests(unittest.TestCase):
    def audit(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(rows))
            return audit_manifest(path, **kwargs)

    def sample(self, **kwargs):
        return dict(dataset_id="synthetic", sample_id="a", patient_id="p", split="train", image_paths=["a.png"], **kwargs)

    def test_missing_note_is_explicit(self):
        sample = ClinicalSample.from_dict(self.sample())
        self.assertFalse(sample.modality_availability["note"])
        self.assertEqual(self.audit([self.sample()])["missing_notes"], 1)

    def test_reject_patient_leakage(self):
        second = self.sample()
        second.update(sample_id="b", split="test", image_paths=["b.png"])
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.audit([self.sample(), second])

    def test_reject_repeated_image(self):
        second = self.sample()
        second.update(sample_id="b", patient_id="q")
        with self.assertRaisesRegex(ValueError, "repeated image"):
            self.audit([self.sample(), second])

    def test_reject_unknown_patient(self):
        row = self.sample(); row["patient_id"] = ""
        with self.assertRaises(ValueError): self.audit([row])

    def test_reject_asset_outside_root(self):
        row = self.sample(); row["image_paths"] = ["../outside.png"]
        with self.assertRaisesRegex(ValueError, "escapes"):
            self.audit([row], check_files=True)

    def test_aggregate_report_has_no_notes(self):
        report = self.audit([self.sample(clinical_note="PRIVATE_SENTINEL")])
        self.assertNotIn("PRIVATE_SENTINEL", json.dumps(report))
