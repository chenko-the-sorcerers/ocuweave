import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from native_classification_data import export, audit
from native_classification_evaluate import metrics, auc, evaluate
from native_data import sha256


class NativeClassification(unittest.TestCase):
    def fixture(self, root):
        records = []
        for split, offset in [("train", 0), ("validation", 10)]:
            for k, code in enumerate("ADGN"):
                for number in range(2 if split == "train" else 1):
                    stem = f"{offset+number}_{code}"
                    p = root / (stem + ".png")
                    image = Image.new(
                        "RGB",
                        (20, 16),
                        (30 + k * 50, 70 + offset + number, 20 + k * 30),
                    )
                    image.save(p)
                    records.append(
                        {
                            "sample_id": "train-" + stem,
                            "image": p.name,
                            "image_sha256": sha256(p),
                            "pixel_sha256": sha256(p),
                            "similarity_group": stem,
                            "disease_stratum": code,
                            "official_split": "train",
                            "split": split,
                        }
                    )
        path = root / "manifest.json"
        path.write_text(json.dumps({"records": records}))
        return path

    def test_probability_metric_oracles(self):
        self.assertEqual(auc([0.5, 0.5], [0, 1]), 0.5)
        self.assertEqual(auc([0.1, 0.9], [0, 1]), 1)
        self.assertEqual(auc([0.9, 0.1], [0, 1]), 0)
        self.assertIsNone(auc([0.1, 0.2], [1, 1]))
        r = metrics([[0.9, 0.1], [0.3, 0.7]], [0, 1], ["disease", "normal"])
        self.assertEqual(r["accuracy"], 1)
        self.assertEqual(r["macro_f1"], 1)
        self.assertEqual(r["dataset_abnormality_triage"]["sensitivity"], 1)
        with self.assertRaises(ValueError):
            metrics([[0.8, 0.8]], [0], ["a", "b"])

    def test_export_label_and_group_guards(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.fixture(root)
            meta = export(root, manifest, root / "export", 16)
            self.assertEqual(meta["partitions"]["train"]["count"], 8)
            self.assertEqual(meta["partitions"]["validation"]["count"], 4)
            self.assertFalse(meta["masks_used_as_labels_or_inputs"])
            data = json.loads(manifest.read_text())
            data["records"][-1]["similarity_group"] = data["records"][0][
                "similarity_group"
            ]
            with self.assertRaises(ValueError):
                audit(data)
            data = json.loads(manifest.read_text())
            data["records"][0]["disease_stratum"] = "H"
            with self.assertRaises(ValueError):
                audit(data)

    @unittest.skipUnless(
        os.environ.get("OCUWEAVE_NATIVE_BUILD"), "Native build required"
    )
    def test_native_train_validation_and_recount(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            manifest = self.fixture(root)
            data = root / "data"
            export(root, manifest, data, 16)
            run = root / "run"
            binary = Path(os.environ["OCUWEAVE_NATIVE_BUILD"]) / "ocuweave-classify"
            args = [
                str(binary),
                "--train",
                str(data / "train.bin"),
                "--train-labels",
                str(data / "train.tsv"),
                "--validation",
                str(data / "validation.bin"),
                "--validation-labels",
                str(data / "validation.tsv"),
                "--output",
                str(run),
                "--width",
                "4",
                "--steps",
                "4",
                "--batch-size",
                "4",
            ]
            subprocess.run(args, check=True, capture_output=True, text=True)
            report = evaluate(data, run, root / "evaluation")
            self.assertEqual(report["results"]["validation"]["samples"], 4)
            self.assertFalse(report["test_evaluated"])
            self.assertEqual(
                sum(
                    map(
                        sum,
                        report["results"]["validation"][
                            "confusion_true_rows_predicted_columns"
                        ],
                    )
                ),
                4,
            )
            changed = args.copy()
            changed[changed.index("--output") + 1] = str(root / "forbidden")
            changed[changed.index("--validation") + 1] = str(data / "train.bin")
            result = subprocess.run(changed, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
