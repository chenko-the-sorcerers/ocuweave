"""Exercise installed-command dispatch on a synthetic OTFID-format archive."""

import json
import tempfile
import unittest
from pathlib import Path
import torch
from ocuweave.cli import parser, dispatch
from ocuweave.demo import create_shapes


class TaskCLI(unittest.TestCase):
    def test_otfid_commands_and_config_guards(self):
        old = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                manifest, archive = create_shapes(root / "data")

                def run(*args):
                    return dispatch(parser().parse_args(list(map(str, args))))

                self.assertIn("software_version", run("doctor"))
                config = root / "config.json"
                run("task-config", "--output", config, "--size", 32, "--steps", 2)
                with self.assertRaises(FileExistsError):
                    run("task-config", "--output", config)
                value = json.loads(config.read_text())
                value.update(width=8, dim=16, queries=2, batch_size=2, save_every=1)
                config.write_text(json.dumps(value))
                common = [
                    "--task",
                    "otfid",
                    "--manifest",
                    manifest,
                    "--source",
                    archive,
                    "--device",
                    "cpu",
                    "--allow-unverified-patients",
                ]
                run(
                    "diagnose",
                    *common,
                    "--config",
                    config,
                    "--output",
                    root / "diagnostic",
                )
                result = run(
                    "fit", *common, "--config", config, "--output", root / "run"
                )
                self.assertTrue(result["complete"])
                report = run(
                    "score",
                    *common,
                    "--checkpoint",
                    root / "run/last.pt",
                    "--output",
                    root / "score",
                )
                self.assertIn("lesion", report["metrics"])
                prediction = run(
                    "infer",
                    "--checkpoint",
                    root / "run/last.pt",
                    "--image",
                    root / "data/example.png",
                    "--output",
                    root / "prediction",
                )
                self.assertEqual(prediction["classes"], ["lesion"])
        finally:
            torch.set_num_threads(old)


if __name__ == "__main__":
    unittest.main()
