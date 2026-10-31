"""Deterministic geometric fixtures exercising the real pipeline, without patient data."""
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import torch

from ocuweave.data.adapters.otfid import audit_otfid
from ocuweave.evaluation.original_grid import evaluate_original
from ocuweave.inference.segmentation import save_prediction
from ocuweave.training.engine import train, validate_config
from ocuweave.utils.provenance import write_json


def _png(array):
    stream = io.BytesIO()
    Image.fromarray(array).save(stream, format='PNG')
    return stream.getvalue()


def create_shapes(output, seed=42):
    """Create a tiny ZIP with the adapter's layout; explicitly label it synthetic."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    labels = ['image,label']
    with zipfile.ZipFile(output / 'shapes.zip', 'x', compression=zipfile.ZIP_DEFLATED) as archive:
        for index in range(24):
            height, width = (48, 64) if index % 3 else (64, 48)
            pixels = rng.integers(10, 90, (height, width, 3), dtype=np.uint8)
            yy, xx = np.mgrid[:height, :width]
            cx, cy = rng.integers(12, width - 12), rng.integers(12, height - 12)
            mask = ((xx - cx) ** 2 / 100 + (yy - cy) ** 2 / 64 <= 1) if index % 2 else np.zeros((height, width), bool)
            pixels[mask] = [215, 170, 100]
            name = f'synthetic_{index:03}'
            labels.append(f'{name}.png,{"active" if mask.any() else "healthy"}')
            # Fixed ZIP timestamps keep archive hashes stable across repeated runs.
            for member, data in [(f'images/{name}.png', _png(pixels)),
                                 (f'masks/{name}-a.png', _png(mask.astype(np.uint8) * 255))]:
                archive.writestr(zipfile.ZipInfo(member, (2026, 1, 1, 0, 0, 0)), data)
        archive.writestr(zipfile.ZipInfo('dataset_labels.csv', (2026, 1, 1, 0, 0, 0)), '\n'.join(labels))
    audit_otfid(output / 'shapes.zip', output / 'audit', seed=seed, allow_unverified_patients=True)
    manifest_path = output / 'audit/manifest.json'
    manifest = json.loads(manifest_path.read_text())
    manifest.update(dataset='synthetic-geometric-shapes', release='1', source_url=None,
                    license='MIT', synthetic=True, audit_protocol='synthetic-shapes-v1')
    for row in manifest['records']:
        row['source'] = 'synthetic-geometric-shapes'
    write_json(manifest_path, manifest)
    audit_path = output / 'audit/audit.json'
    audit = json.loads(audit_path.read_text())
    audit.update(synthetic=True, audit_protocol='synthetic-shapes-v1',
                 interpretation='Adapter-format fixture. Active/healthy strings are format tokens, not clinical labels.')
    write_json(audit_path, audit)
    # Use a validation image for the demo prediction; the test split remains untouched.
    sample = next(row for row in manifest['records'] if row['split'] == 'validation' and row['lesion_pixels'] > 0)
    with zipfile.ZipFile(output / 'shapes.zip') as archive:
        (output / 'example.png').write_bytes(archive.read(sample['image']))
        (output / 'reference.png').write_bytes(archive.read(sample['masks'][0]))
    return manifest_path, output / 'shapes.zip'


def run_demo(output, *, architecture='query_detail', seed=42, epochs=2, device='cpu'):
    config = dict(architecture=architecture, width=8, dim=16, queries=4,
                  size=64, batch_size=8, epochs=epochs, learning_rate=0.001, seed=seed)
    validate_config(config)
    output = Path(output)
    manifest, archive = create_shapes(output, seed)
    write_json(output / 'config.json', config)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        checkpoint = train(manifest, archive, output / 'run', config,
                           allow_unverified=True, device=device)
        evaluation = evaluate_original(manifest, archive, checkpoint, output / 'validation.json',
                                       allow_unverified=True, device=device)
        save_prediction(checkpoint, output / 'example.png', output / 'prediction', device=device)
    finally:
        torch.set_num_threads(previous_threads)
    summary = {'synthetic': True, 'purpose': 'Installation and pipeline smoke test, not a research benchmark',
               'architecture': architecture, 'checkpoint': str(checkpoint), 'test_split_evaluated': False,
               'validation_samples': evaluation['metrics']['samples'],
               'outputs': ['run/best.pt', 'run/history.json', 'run/provenance.json',
                           'validation.json', 'prediction/mask.png', 'prediction/probability.npy',
                           'prediction/prediction.json']}
    write_json(output / 'demo.json', summary)
    return summary
