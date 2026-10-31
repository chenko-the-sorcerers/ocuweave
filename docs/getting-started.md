# Getting started

## Environment and installation

Use Python 3.10 or later in a virtual environment. Install the repository with `python -m pip install -e ".[research]"`. This installs PyTorch, NumPy, Pillow and Matplotlib. `pip install .` alone installs the lightweight metadata/audit package and CLI help, not the neural-network dependencies.

Run `ocuweave --help`, then `ocuweave demo --output outputs/demo --device cpu`. `python -m ocuweave` exposes the same subcommands. Run `ocuweave COMMAND --help` for arguments. All output paths are local unless you deliberately point them at a mounted Drive folder. No command mounts Drive, authorizes services, purchases compute, or downloads model weights.

The synthetic demo uses the actual audit, training, checkpoint, original-grid evaluation and inference modules. It generates simple ellipses and empty masks. It is an installation smoke test and is not evidence of retinal segmentation performance. Check `demo.json`, `run/history.json`, `run/checkpoint.json`, `validation.json` and `prediction/prediction.json` to verify completion.

## Real-data workflow

1. Obtain OTFID v3 from its [official record](https://zenodo.org/records/13883485), following the dataset license. Save the unchanged archive locally as `data/otfid.zip`. The source checkout also includes `python scripts/download_data.py --help` for the acquisition workflow; do not assume a synthetic example is medical data.
2. Audit the archive and paired masks. If independently verified patient identifiers are available, supply a JSON map of image stem to patient ID with `--patient-map`. Otherwise explicitly acknowledge exploratory image grouping:

```bash
ocuweave prepare-otfid --archive data/otfid.zip --output data/audit \
  --allow-unverified-patients
```

3. Inspect `data/audit/audit.json`, the retained/excluded records, label contradictions, duplicate groups, and split membership. Automated technical validity does not establish clinical correctness. Quality-flagged images require documented review, not automatic deletion.
4. For the project’s existing OTFID cohort, do **not** substitute a freshly generated split for the historical or controlled manifest. Follow [development freezing](protocols/exploratory_training.md), review the known near-duplicate candidate, and retain the original test split. Advanced tools remain available as `scripts/freeze_development.py --help` and `scripts/build_review_bundle.py --help` from a source checkout.
5. Write a configuration, train, then evaluate validation only:

```bash
ocuweave init-config --architecture query --size 192 --seed 42 --output outputs/query192.json
ocuweave train --manifest data/audit/manifest.json --archive data/otfid.zip \
  --config outputs/query192.json --output outputs/query192 \
  --device cuda --allow-unverified-patients
ocuweave evaluate --manifest data/audit/manifest.json --archive data/otfid.zip \
  --checkpoint outputs/query192/best.pt --output outputs/query192-validation.json \
  --device cuda --allow-unverified-patients
```

The generated config records width 24, dimension 48, 8 queries, batch 8, 20 epochs and learning rate 0.001. Fixed engine settings are AdamW weight decay 0.0001, BCE + Dice, gradient clipping 1 and cosine scheduling. See [mathematical specification](https://data-sorcerers.com/ocuweave/#mathematical-formulation). Change one comparison factor at a time; the separate controlled runner records the nine-run protocol. Training rejects a mismatched archive or cross-split group leakage.

### Evaluation semantics

The installed CLI defaults to `--split validation --grid original`: remove letterbox padding, restore probabilities bilinearly to the original dimensions, then threshold at 0.5. Reports include per-image counts, aggregate Dice/IoU, small-mask strata and healthy false-positive area. An empty stratum is unavailable, not zero. Both masks empty gives Dice/IoU 1; only one empty gives 0.

`--grid input` selects the historical input-grid evaluator. Only that legacy evaluator accepts explicit `--split test` in this release. Do not compare its scores directly with original-grid scores or use the old test set to select changes. Source scripts retain their existing defaults for notebook compatibility; `scripts/evaluate.py` defaults to the legacy test evaluator, unlike the new CLI. Prefer the CLI for development.

### Prediction

```bash
ocuweave predict --checkpoint outputs/query192/best.pt --image /path/to/image.png \
  --output outputs/prediction --device cpu
```

Outputs are `probability.npy`, `mask.png` and `prediction.json`, with original geometry and input/checkpoint hashes. Use checkpoints you trained or obtained from a trusted source. No public pretrained weight is part of this release. The model predicts binary foreground according to its training targets; it does not assign lesion activity, glaucoma stage, blood pressure or prognosis.

## Compute and recovery

Use `--device cpu` for the tiny demo. Real 384-pixel training is better run on your available Colab GPU. `--device auto` chooses CUDA, then MPS, then CPU; an explicit unavailable device fails instead of silently changing the experiment. Reduce batch size only in a separately recorded experiment; do not quietly change the locked comparison settings.

Runs checkpoint the best validation model and write history after each epoch. This release does **not** implement full optimizer/scheduler resume. If Colab disconnects, preserve the partial directory and start a new run directory; a best-model checkpoint alone is not a reproducible continuation. Checkpoints and reports should live under a persistent mounted Drive path for Colab.

## Troubleshooting

| Symptom | Action |
|---|---|
| `ocuweave: command not found` | Activate the environment or use `python -m ocuweave` with its interpreter. |
| Missing torch / NumPy / Pillow | Install the research extra in that environment. |
| Output already exists | Keep the prior artifacts; choose a new directory/file. |
| Archive or manifest mismatch | Use the exact audited archive and checkpoint-compatible manifest. |
| Unknown patient guard | Verify a patient map, or explicitly label the run exploratory using the flag. |
| Zero Dice / IoU | Inspect the original/reference/prediction pairs and TP/FP/FN; do not replace the score. |
| No GPU / quota exhausted | Run CPU audits and the synthetic smoke test; leave research runs pending. |

## Unified task workflows

For installed-package FIVES/IDRiD/OTFID commands, interrupted-run resume, training-only diagnostics and optional tiled inference, use [Task workflows](task-workflows.md). The new fixed-step protocol is separate from the historical epoch-based examples above.
