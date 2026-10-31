import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from ocuweave.data.auditing.clinical_notes import audit_notes, note_flags
from ocuweave.data.auditing.fairvlmed_images import audit_images, inspect_npz


def payload(image, label=1, **extra):
    stream = io.BytesIO()
    np.savez_compressed(stream, slo_fundus=image, glaucoma=np.array(label), **extra)
    return stream.getvalue()


class DeepAuditTests(unittest.TestCase):
    def test_objects_are_never_decoded(self):
        image = np.arange(256, dtype=np.uint8).reshape(16, 16)
        result = inspect_npz(payload(image, note=np.array([{"secret": "not loaded"}], dtype=object)))
        self.assertFalse(result["constant"])
        with self.assertRaises(ValueError):
            inspect_npz(payload(np.array([[{}] * 16] * 16, dtype=object)))
        with self.assertRaises(ValueError):
            inspect_npz(payload(np.full((16, 16), np.nan)))
        with self.assertRaises(ValueError):
            inspect_npz(payload(image, label=2))

    def test_note_flags_do_not_claim_semantics_or_read_test_notes(self):
        flags = note_flags("No family history of glaucoma; glaucoma suspect on timolol.")
        for key in ("diagnosis_term", "negation_context", "family_context", "suspect_context", "treatment_proxy"):
            self.assertTrue(flags[key])
        rows = [{"filename": "a", "use": "training", "note": "Private glaucoma text"},
                {"filename": "b", "use": "validation", "note": " PRIVATE  glaucoma TEXT "},
                {"filename": "c", "use": "test"}]
        report = audit_notes(rows)
        self.assertEqual(report["cross_development_split_duplicate_groups"], 1)
        self.assertNotIn("Private glaucoma text", json.dumps(report))
        self.assertFalse(report["semantic_review_complete"])
        self.assertEqual(len(report["review_queue"]), 2)

    def test_duplicate_label_and_corrupt_file_are_reported_not_silently_removed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Dataset").mkdir(); (root / "ReadMe").mkdir()
            with (root / "ReadMe/data_summary.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["filename", "use", "note", "glaucoma"])
                writer.writeheader()
                for i, split in enumerate(("training", "validation", "test")):
                    writer.writerow(dict(filename=f"{i}.npz", use=split, note="private", glaucoma="no"))
            image = np.arange(256, dtype=np.uint8).reshape(16, 16)
            with zipfile.ZipFile(root / "Dataset/dataset.zip", "w") as archive:
                archive.writestr("0.npz", payload(image, label=0))
                archive.writestr("1.npz", payload(image, label=1))
                archive.writestr("2.npz", b"broken")
            report = audit_images(root, root / "audit")
            self.assertEqual((report["decoded"], report["failed"]), (2, 1))
            self.assertEqual(report["cross_split_exact_pixel_groups"], 1)
            self.assertEqual(report["label_mismatches"], 1)
            with self.assertRaises(FileExistsError):
                audit_images(root, root / "audit")
