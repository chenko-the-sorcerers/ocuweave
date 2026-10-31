import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from ocuweave.demo_tasks import create_task_fixture
from ocuweave.training.task_config import default_config
from ocuweave.training.tasks import fit, step_indices
from ocuweave.evaluation.tasks import evaluate_task
from ocuweave.inference.tasks import infer
from ocuweave.diagnostics.readiness import diagnostics
from ocuweave.data.tasks import SegmentationTask
from ocuweave.losses.observed import observed_components
from ocuweave.data.preprocessing.tiles import tile_boxes, merge_tiles
from ocuweave.data.augmentation.paired import horizontal_flip


class TaskWorkflows(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.config = {
            **default_config(),
            "width": 8,
            "dim": 16,
            "queries": 2,
            "size": 32,
            "steps": 4,
            "batch_size": 2,
            "save_every": 1,
        }

    def test_exact_cpu_resume_and_validation_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            manifest = create_task_fixture("fives", p / "data")
            args = ("fives", manifest, p / "data")
            full = fit(
                *args, p / "full", self.config, device="cpu", allow_unverified=True
            )
            partial = fit(
                *args,
                p / "partial",
                self.config,
                device="cpu",
                allow_unverified=True,
                max_steps_this_session=2,
            )
            self.assertFalse(partial["complete"])
            with self.assertRaisesRegex(ValueError, "Interrupted"):
                evaluate_task(
                    *args, partial["checkpoint"], p / "not-final", allow_unverified=True
                )
            resumed = fit(
                *args,
                p / "resumed",
                self.config,
                device="cpu",
                allow_unverified=True,
                resume=partial["checkpoint"],
            )
            a = torch.load(full["checkpoint"], weights_only=True)
            b = torch.load(resumed["checkpoint"], weights_only=True)
            for k in a["model"]:
                torch.testing.assert_close(a["model"][k], b["model"][k], rtol=0, atol=0)
            self.assertEqual(a["history"], b["history"])
            score = evaluate_task(
                *args, resumed["checkpoint"], p / "evaluation", allow_unverified=True
            )
            self.assertFalse(score["test_evaluated"])
            self.assertEqual(score["split"], "validation")
            row = next(
                r
                for r in json.loads(manifest.read_text())["records"]
                if r["split"] == "validation"
            )
            prediction = infer(
                resumed["checkpoint"],
                p / "data" / row["image"],
                p / "prediction",
                tile_size=24,
                overlap=8,
            )
            self.assertEqual(prediction["shape"], [1, 32, 48])
            self.assertGreater(prediction["tile_count"], 1)
            with self.assertRaises(FileExistsError):
                infer(
                    resumed["checkpoint"], p / "data" / row["image"], p / "prediction"
                )
            wrong = {**self.config, "steps": 5}
            with self.assertRaisesRegex(ValueError, "exact configuration"):
                fit(
                    *args,
                    p / "wrong",
                    wrong,
                    device="cpu",
                    allow_unverified=True,
                    resume=partial["checkpoint"],
                )

    def test_idrid_unknown_labels_and_train_only_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            manifest = create_task_fixture("idrid", p / "data")
            data = json.loads(manifest.read_text())
            with self.assertRaisesRegex(ValueError, "allow-unverified"):
                SegmentationTask("idrid", manifest, p / "data")
            val = next(
                r["sample_id"] for r in data["records"] if r["split"] == "validation"
            )
            train = next(
                r["sample_id"] for r in data["records"] if r["split"] == "train"
            )
            cfg = {**self.config, "steps": 2}
            with self.assertRaisesRegex(ValueError, "Only training"):
                diagnostics(
                    "idrid",
                    manifest,
                    p / "data",
                    p / "bad",
                    cfg,
                    device="cpu",
                    allow_unverified=True,
                    sample_ids=[train, val],
                )
            self.assertFalse((p / "bad").exists())
            report = diagnostics(
                "idrid",
                manifest,
                p / "data",
                p / "diag",
                cfg,
                device="cpu",
                allow_unverified=True,
            )
            self.assertFalse(report["gpu_executed"])
            self.assertFalse(report["validation_evaluated"])
            self.assertTrue(report["checkpoint_reload_matches"])
            self.assertTrue(
                set(report["sample_ids"])
                <= set(r["sample_id"] for r in data["records"] if r["split"] == "train")
            )
            self.assertTrue(
                any(
                    not row["annotation_available"] and row["metrics"] is None
                    for row in report["final"]
                )
            )
            history = json.loads((p / "diag/history.json").read_text())
            self.assertGreater(
                history[0]["parameter_groups"]["heads.0"]["gradient_l2"], 0
            )
            self.assertGreater(
                history[0]["parameter_groups"]["local_detail"]["update_l2"], 0
            )
            run = fit(
                "idrid",
                manifest,
                p / "data",
                p / "run",
                cfg,
                device="cpu",
                allow_unverified=True,
            )
            score = evaluate_task(
                "idrid",
                manifest,
                p / "data",
                run["checkpoint"],
                p / "val",
                allow_unverified=True,
            )
            self.assertEqual(set(score["per_class"]), {"MA", "HE", "EX", "SE"})
            data["records"][0]["image"] = "../outside.jpg"
            manifest.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "relative"):
                SegmentationTask("idrid", manifest, p / "data", allow_unverified=True)

    def test_observed_loss_oracle_and_missing_gradient(self):
        logits = torch.zeros(1, 2, 2, 3, requires_grad=True)
        target = torch.ones_like(logits)
        valid = torch.ones(1, 1, 2, 3)
        valid[:, :, :, -1] = 0
        target.data[:, 0, :, -1] = float("nan")
        target.data[:, 1] = float("nan")
        observed = torch.tensor([[True, False]])
        components = observed_components(logits, target, valid, observed)
        self.assertAlmostEqual(float(components["bce"].detach()), np.log(2), places=6)
        self.assertAlmostEqual(float(components["soft_dice"].detach()), 2 / 7, places=6)
        components["total"].backward()
        self.assertEqual(float(logits.grad[:, 1].abs().sum()), 0)
        self.assertEqual(float(logits.grad[:, :, :, -1].abs().sum()), 0)
        self.assertGreater(float(logits.grad[:, 0, :, :2].abs().sum()), 0)

    def test_tiling_reconstruction_and_geometry_guards(self):
        original = np.linspace(0, 1, 37 * 53, dtype=np.float32).reshape(1, 37, 53)
        boxes = tile_boxes(37, 53, 16, 5)
        tiles = [original[:, y:yy, x:xx] for y, x, yy, xx in boxes]
        np.testing.assert_allclose(
            merge_tiles(tiles, boxes, (37, 53)), original, rtol=0, atol=1e-7
        )
        with self.assertRaisesRegex(ValueError, "uncovered"):
            merge_tiles(tiles[:1], boxes[:1], (37, 53))
        with self.assertRaises(ValueError):
            tile_boxes(37, 53, 16, 16)
        self.assertEqual(tile_boxes(5, 7, 16), [(0, 0, 5, 7)])

    def test_pairing_and_sample_order_are_independent_of_model_rng(self):
        x = torch.arange(24.0).reshape(2, 1, 3, 4)
        batch = {
            "image": x,
            "mask": x.clone(),
            "valid": torch.ones_like(x),
            "available": torch.ones(2, 1, dtype=torch.bool),
        }
        actual = horizontal_flip(batch, seed=4, probability=1)
        torch.testing.assert_close(actual["image"], x.flip(-1))
        torch.testing.assert_close(actual["image"], actual["mask"])
        torch.testing.assert_close(batch["image"], x)
        epoch = sum([step_indices(5, 2, 42, i) for i in range(3)], [])
        self.assertEqual(sorted(epoch), list(range(5)))


if __name__ == "__main__":
    unittest.main()
