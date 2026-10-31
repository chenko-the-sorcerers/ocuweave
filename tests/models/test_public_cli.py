"""Public package behavior, real synthetic round trips and immutable artifacts."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from ocuweave.demo import run_demo, create_shapes
from ocuweave.inference.segmentation import predict_image, save_prediction
from ocuweave.utils.provenance import sha256, software_provenance
from ocuweave import __version__


class PublicWorkflowTests(unittest.TestCase):
    def test_demo_round_trip_for_every_architecture(self):
        with tempfile.TemporaryDirectory() as directory:
            for architecture in ('query', 'query_detail', 'fcn'):
                root = Path(directory) / architecture
                summary = run_demo(root, architecture=architecture, epochs=1)
                self.assertTrue(summary['synthetic'])
                self.assertFalse(summary['test_split_evaluated'])
                evaluation = json.loads((root / 'validation.json').read_text())
                self.assertEqual(evaluation['split'], 'validation')
                manifest = json.loads((root / 'audit/manifest.json').read_text())
                self.assertEqual(manifest['dataset'], 'synthetic-geometric-shapes')
                self.assertIsNone(manifest['source_url'])
                probability = np.load(root / 'prediction/probability.npy')
                with Image.open(root / 'example.png') as image:
                    self.assertEqual(probability.shape, (image.height, image.width))
                    expected = predict_image(root / 'run/best.pt', image)
                np.testing.assert_allclose(probability, expected, atol=1e-6)
                with Image.open(root / 'prediction/mask.png') as image:
                    np.testing.assert_array_equal(np.asarray(image) > 0, probability >= 0.5)
                sample_id = next(r['sample_id'] for r in manifest['records']
                                 if r['split'] == 'validation' and r['lesion_pixels'] > 0)
                row = next(r for r in evaluation['per_image'] if r['sample_id'] == sample_id)
                with Image.open(root / 'reference.png') as image:
                    reference = np.asarray(image) > 0
                self.assertEqual(row['tp'], int(((probability >= 0.5) & reference).sum()))
                self.assertEqual(row['fp'], int(((probability >= 0.5) & ~reference).sum()))
                checkpoint_hash = sha256(root / 'run/best.pt')
                with self.assertRaises(FileExistsError):
                    run_demo(root)
                with self.assertRaises(FileExistsError):
                    save_prediction(root / 'run/best.pt', root / 'example.png', root / 'prediction')
                self.assertEqual(checkpoint_hash, sha256(root / 'run/best.pt'))

    def test_synthetic_archive_determinism_and_config_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = create_shapes(root / 'a', seed=42)
            _, second = create_shapes(root / 'b', seed=42)
            self.assertEqual(sha256(first), sha256(second))
            with self.assertRaises(ValueError):
                run_demo(root / 'invalid', epochs=0)
            self.assertFalse((root / 'invalid').exists())

    def test_installed_module_cli_and_validation_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, '-m', 'ocuweave']
            result = subprocess.run(command + ['--help'], cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            path = str(Path(directory) / 'config.json')
            args = command + ['init-config', '--output', path, '--size', '384']
            self.assertEqual(subprocess.run(args, cwd=directory, capture_output=True).returncode, 0)
            self.assertEqual(json.loads(Path(path).read_text())['architecture'], 'query_detail')
            self.assertNotEqual(subprocess.run(args, cwd=directory, capture_output=True).returncode, 0)
            result = subprocess.run(command + ['evaluate', '--manifest', 'missing', '--archive', 'missing',
                '--checkpoint', 'missing', '--output', 'unused', '--split', 'test'],
                cwd=directory, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('validation-only', result.stderr)

    def test_provenance_does_not_use_callers_git_repository(self):
        before = software_provenance()
        original = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                after = software_provenance()
            finally:
                os.chdir(original)
        self.assertEqual(before, after)
        self.assertEqual(after['software_version'], __version__)

    def test_training_evaluation_and_prediction_subcommands(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, archive = create_shapes(root / 'data')
            config = dict(architecture='query_detail', width=4, dim=8, queries=2,
                          size=32, batch_size=8, epochs=1, learning_rate=0.001, seed=42)
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config))
            command = [sys.executable, '-m', 'ocuweave']
            shared = ['--manifest', str(manifest), '--archive', str(archive),
                      '--allow-unverified-patients', '--device', 'cpu']
            calls = [
                ['train', *shared, '--config', str(config_path), '--output', str(root / 'run')],
                ['evaluate', *shared, '--checkpoint', str(root / 'run/best.pt'), '--output', str(root / 'metrics.json')],
                ['predict', '--checkpoint', str(root / 'run/best.pt'), '--image', str(root / 'data/example.png'),
                 '--output', str(root / 'prediction')],
            ]
            for args in calls:
                result = subprocess.run(command + args, cwd=directory, capture_output=True, text=True,
                                        env={**os.environ, 'OMP_NUM_THREADS': '1'}, timeout=90)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads((root / 'metrics.json').read_text())['split'], 'validation')
            self.assertTrue((root / 'prediction/mask.png').exists())


if __name__ == '__main__':
    unittest.main()
