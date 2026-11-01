import array
import hashlib
import importlib.util
import json
import os
import subprocess
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from native_data import export, sha256, tile_boxes
from native_evaluate import evaluate
from native_tiles import merge_tiles
from native_patch_programme import run as run_patch_programme
from native_report import summarize


class NativeIOScripts(unittest.TestCase):
    def test_export_restore_and_missingness(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new("RGB", (16, 16), (128, 64, 32)).save(root / "image.png")
            mask = Image.new("L", (16, 16))
            mask.paste(255, (2, 3, 7, 9))
            mask.convert("RGB").save(root / "mask.png")
            row = {
                "sample_id": "sample",
                "split": "train",
                "image": "image.png",
                "mask": "mask.png",
                "image_sha256": sha256(root / "image.png"),
                "mask_sha256": sha256(root / "mask.png"),
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": [row]}))
            meta = export(root, manifest, root / "input.bin", ["sample"], 16, "fives")
            self.assertEqual(
                meta["samples"][0]["letterbox"],
                {"width": 16, "height": 16, "left": 0, "top": 0},
            )
            with (root / "prob.bin").open("wb") as f:
                f.write(b"OWPROB01" + struct.pack("<4I", 1, 1, 16, 16))
                values = array.array("f", (1.0 if x else 0.0 for x in mask.tobytes()))
                if sys.byteorder != "little":
                    values.byteswap()
                values.tofile(f)
            report = evaluate(
                root / "prob.bin", root / "input.json", root, root / "result"
            )
            self.assertEqual(report["summary"]["vessel"]["positive_macro_dice"], 1)
            self.assertEqual(report["rows"][0]["predicted_area_pixels"], 30)
            self.assertEqual(
                report["rows"][0]["predicted_image_area_percent"], 100 * 30 / 256
            )
            row["split"] = "test"
            manifest.write_text(json.dumps({"records": [row]}))
            with self.assertRaises(ValueError):
                export(root, manifest, root / "forbidden.bin", ["sample"], 16, "fives")
            row["split"] = "train"
            row["mask_sha256"] = "0" * 64
            manifest.write_text(json.dumps({"records": [row]}))
            with self.assertRaises(ValueError):
                export(root, manifest, root / "changed.bin", ["sample"], 16, "fives")

    def test_otfid_decoded_hash_contract(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            image = Image.new("RGB", (16, 16), (8, 12, 14))
            image.save(root / "image.png")
            mask = Image.new("L", (16, 16))
            mask.paste(255, (2, 2, 5, 5))
            mask.save(root / "mask.png")
            raw = bytes(bool(v) for v in mask.tobytes())
            row = {
                "sample_id": "tox",
                "split": "train",
                "image": "image.png",
                "masks": ["mask.png"],
                "pixel_sha256": hashlib.sha256(
                    str((16, 16, 3)).encode() + image.tobytes()
                ).hexdigest(),
                "mask_sha256": hashlib.sha256(raw).hexdigest(),
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": [row]}))
            report = export(root, manifest, root / "otfid.bin", ["tox"], 16, "otfid")
            self.assertEqual(report["classes"], ["lesion"])

    @unittest.skipUnless(
        os.environ.get("OCUWEAVE_NATIVE_BUILD"),
        "Standalone executable integration requires a native build",
    )
    def test_patch_training_and_whole_image_inference(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init", "-q", str(repo)], check=True)
            subprocess.run(
                [
                    "git",
                    "-c",
                    "user.name=Native test",
                    "-c",
                    "user.email=test@example.invalid",
                    "commit",
                    "--allow-empty",
                    "-qm",
                    "Fixture",
                ],
                cwd=repo,
                check=True,
            )
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True
            ).strip()
            data, session = root / "data", root / "session"
            data.mkdir()
            session.mkdir()
            Image.new("RGB", (32, 32), (128, 64, 32)).save(data / "image.png")
            mask = Image.new("L", (32, 32))
            mask.paste(255, (4, 4, 9, 9))
            mask.save(data / "mask.png")
            manifest = data / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "records": [
                            {
                                "sample_id": "sample",
                                "split": "train",
                                "image": "image.png",
                                "mask": "mask.png",
                                "image_sha256": sha256(data / "image.png"),
                                "mask_sha256": sha256(data / "mask.png"),
                            }
                        ]
                    }
                )
            )
            export(data, manifest, session / "fives.bin", ["sample"], 16, "fives")
            build = Path(os.environ["OCUWEAVE_NATIVE_BUILD"]).resolve()
            subprocess.run(
                [
                    str(build / "ocuweave-lqnet"),
                    "--batch",
                    str(session / "fives.bin"),
                    "--classes",
                    "1",
                    "--output",
                    str(session / "fives-baseline"),
                    "--steps",
                    "1",
                ],
                check=True,
                capture_output=True,
            )
            (session / "programme-status.json").write_text(
                json.dumps({"status": "completed", "runs": []})
            )
            (session / "patch-contract.json").write_text(
                json.dumps(
                    {
                        "source_commit": revision,
                        "manifest_sha256": sha256(manifest),
                        "ids": ["sample"],
                        "input_size": 16,
                        "patch_size": 16,
                        "inference_overlap": 4,
                        "steps": 1,
                        "device": "cpu",
                    }
                )
            )
            report = run_patch_programme(repo, build, session, data, manifest)
            self.assertEqual(len(report["rows"]), 2)
            self.assertEqual(
                json.loads((session / "patch-status.json").read_text())["status"],
                "completed",
            )
            for row in report["rows"]:
                self.assertEqual(row["image_area_pixels"], 1024)
                self.assertEqual(row["metrics"]["reference_area_pixels"], 25)

    def test_report_preserves_pending_and_recounts_metrics(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "contract-locked.json").write_text(
                json.dumps({"source_commit": "fixture", "runs": [{"name": "baseline"}]})
            )
            state = {
                "source_commit": "fixture",
                "runs": [
                    {
                        "task": "fives",
                        "name": "baseline",
                        "status": "completed",
                        "checkpoint_sha256": "checkpoint",
                        "batch_sha256": "batch",
                    }
                ],
            }
            (root / "programme-status.json").write_text(json.dumps(state))
            run = root / "fives-baseline"
            (run / "original-grid").mkdir(parents=True)
            (run / "status.json").write_text(
                json.dumps({"checkpoint_reload_matches": True, "elapsed_seconds": 1})
            )
            (run / "protocol.json").write_text(
                json.dumps(
                    {
                        "source_commit": "fixture",
                        "source_dirty": False,
                        "parameters": 12,
                        "start_step": 0,
                        "additional_steps": 200,
                    }
                )
            )
            (run / "metrics.json").write_text(
                json.dumps(
                    {
                        "rows": [
                            {
                                "class_index": 0,
                                "annotation_available": True,
                                "cldice": 0.75,
                            }
                        ]
                    }
                )
            )
            metrics = {
                "summary": {"vessel": {"positive_macro_dice": 0.5}},
                "rows": [
                    {
                        "class": "vessel",
                        "image_area_pixels": 100,
                        "metrics": {
                            "tp": 2,
                            "fp": 1,
                            "fn": 3,
                            "dice": 0.5,
                            "iou": 1 / 3,
                        },
                    }
                ],
            }
            path = run / "original-grid/metrics.json"
            path.write_text(json.dumps(metrics))
            (root / "fives-baseline-memory.json").write_text(
                json.dumps({"samples": []})
            )
            report = summarize(root, root / "report")
            self.assertFalse(report["core_complete"])
            self.assertEqual(report["runs"][0]["status"], "queued")
            row = report["class_results"][0]
            self.assertAlmostEqual(row["pooled_precision"], 2 / 3)
            self.assertAlmostEqual(row["pooled_recall"], 0.4)
            self.assertEqual(row["false_positive_image_area_percent"], 1)
            metrics["rows"][0]["metrics"]["dice"] = 0.9
            path.write_text(json.dumps(metrics))
            with self.assertRaises(ValueError):
                summarize(root, root / "tampered")

    def test_overlap_reconstruction_oracle(self):
        W, H = 9, 7
        boxes = tile_boxes(W, H, 4, 2)
        tiles = [
            [float(y * W + x) for y in range(y0, y1) for x in range(x0, x1)]
            for x0, y0, x1, y1 in boxes
        ]
        self.assertEqual(
            list(merge_tiles(tiles, boxes, W, H)), list(map(float, range(W * H)))
        )
        with self.assertRaises(ValueError):
            merge_tiles(tiles[:-1], boxes[:-1], W, H)
        self.assertEqual(tile_boxes(2, 3, 4, 0), [(0, 0, 2, 3)])
