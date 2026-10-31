<div align="center">

# OcuWeave

### Evidence-Grounded Vision–Language Learning with Incomplete Clinical Records for Ophthalmic Disease Assessment

**Marchel Andrian Shevchenko** · Data Sorcerers · 2026

[Research page](https://data-sorcerers.com/ocuweave/) · [Getting started](docs/getting-started.md) · [Method](docs/architecture.md) · [Results](docs/results/README.md) · [Changelog](CHANGELOG.md)

[![Checks](https://github.com/chenko-the-sorcerers/ocuweave/actions/workflows/ci.yml/badge.svg)](https://github.com/chenko-the-sorcerers/ocuweave/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Software v0.10.0](https://img.shields.io/badge/software-v0.10.0-334bd3.svg)](https://github.com/chenko-the-sorcerers/ocuweave/tree/v0.10.0)

![OcuWeave research illustration](docs/assets/ocuweave-thumbnail.webp)

</div>

*Educational AI-generated fundus illustration and lesion overlay. This artwork is not patient data, a reference annotation, or a model prediction.*

OcuWeave is an open research toolkit for ophthalmic segmentation, with ocular toxoplasmosis as its first experimental task. **OcuWeave–LQNet** adds a local-detail residual branch to our compact query segmenter. Both networks are independently implemented and initialized from random weights. PyTorch supplies tensor operations and automatic differentiation.

The current release provides a runnable **image-only** pipeline. The title describes the broader research direction; LLM–EMR fusion and longitudinal prediction are not implemented. The manuscript is in preparation; no conference acceptance or state-of-the-art result is claimed.

## Start here

- **Try the software:** run the synthetic end-to-end demo below; no dataset access or GPU is needed.
- **Reproduce an experiment:** use the [real-data workflow](docs/getting-started.md#real-data-workflow) and [locked development protocol](docs/protocols/exploratory_training.md).
- **Inspect the method:** see [architecture](docs/architecture.md), [equations on the project page](https://data-sorcerers.com/ocuweave/#mathematical-formulation), and the [Python API](docs/api.md).
- **Assess the evidence:** read the [results register](docs/results/README.md), including unsuccessful predictions and unresolved data-quality questions.

## Install

Python **3.10+**. Start with CPU; CUDA is optional. C++ tooling is required only for the separate native example. Installation from source is supported; this release is not published to PyPI.

```bash
git clone --branch v0.10.0 https://github.com/chenko-the-sorcerers/ocuweave.git
cd ocuweave
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[research]"
ocuweave --version
```

For a CUDA-specific PyTorch build, install the appropriate PyTorch distribution for your machine before installing the research extra. Google Colab can run these same commands after cloning; mount your own Drive and put persistent outputs there. Hosted notebooks remain optional launchers.

## A complete CPU smoke test

```bash
ocuweave demo --output outputs/demo --device cpu
```

This command generates **24 geometric-shape images**, audits image–mask pairs, trains the small LQNet variant for two epochs, evaluates validation predictions at original dimensions, and saves a checkpoint and prediction. It does not download medical images, access Drive, or evaluate the test split. The synthetic scores are installation checks, not research results.

```text
outputs/demo/
├── shapes.zip                 # Generated fixture, not OTFID data
├── audit/manifest.json        # Frozen fixture split and archive hash
├── config.json
├── example.png / reference.png
├── run/
│   ├── best.pt                # Validation-selected synthetic checkpoint
│   ├── history.json
│   ├── provenance.json
│   └── checkpoint.json
├── validation.json            # Original-grid Dice, IoU and FP analysis
├── prediction/
│   ├── probability.npy        # Float probabilities at original dimensions
│   ├── mask.png               # Threshold 0.5; values 0/255
│   └── prediction.json        # Input/checkpoint hashes and geometry
└── demo.json
```

Run directories are immutable: choose another output path to repeat a run. `python -m ocuweave` is equivalent to `ocuweave`, including after installation from a wheel outside the source checkout.

## Train, evaluate, predict

The following commands assume an **audited local archive and manifest**. Acquisition, patient-identity constraints, cleaning and split review are documented in [Getting started](docs/getting-started.md). They must precede claims about model performance.

```bash
ocuweave init-config --architecture query_detail --size 384 --seed 42 \
  --output outputs/lqnet-384-seed42.json

ocuweave train --manifest data/audit/manifest.json --archive data/otfid.zip \
  --config outputs/lqnet-384-seed42.json --output outputs/lqnet-384-seed42 \
  --device cuda --allow-unverified-patients

ocuweave evaluate --manifest data/audit/manifest.json --archive data/otfid.zip \
  --checkpoint outputs/lqnet-384-seed42/best.pt --output outputs/validation-384.json \
  --device cuda --allow-unverified-patients

ocuweave predict --checkpoint outputs/lqnet-384-seed42/best.pt \
  --image /path/to/fundus.png --output outputs/prediction --device cpu
```

`--allow-unverified-patients` explicitly permits exploratory image grouping; it does **not** verify independence. Omit the flag for a verified patient-group manifest. Evaluation defaults to **validation on the original image grid**. Prediction requires your own checkpoint: no validated pretrained weights are bundled.

## Method at a glance

![OcuWeave–LQNet: query-mask logits with an additive local-detail residual and original-grid restoration](docs/assets/lqnet-architecture.svg)

| Configuration key | Implementation | Parameters at default dimensions | Experimental role |
|---|---|---:|---|
| `query` | [OcuWeaveSegmenter](src/ocuweave/models/segmenter.py) | 109,754 | Query baseline |
| `query_detail` | [LocalDetailSegmenter](src/ocuweave/models/local_detail.py) | 111,411 | LQNet local-detail ablation |
| `fcn` | [CompactFCN](src/ocuweave/baselines/compact_fcn.py) | Reported per run | Encoder/pixel-feature matched baseline |

The local branch starts with a zero output projection. It refines logits at half input resolution. All variants use masked BCE + soft Dice, AdamW, and explicit seed/configuration records. The [model card](docs/model_cards/ocuweave.md) describes scope and limitations.

## Evidence and reproducibility

- One historical OTFID query-network feasibility pilot is documented. Some validation examples have **zero overlap**. These remain reported failures.
- The controlled matrix—query 192, query 384, LQNet 384; seeds 42, 123 and 2026—has **0/9 completed runs** in the recorded research status. This software release does not complete those experiments.
- The new development manifest is frozen at **197 training / 41 validation**, with all **104 quality-flagged images retained** and all 42 historical test rows preserved. CPU smoke checks passed for all three variants; GPU training remains blocked by Colab usage limits. See the [execution record](docs/experiments/2026-10-31-development-freeze.md) and [continuation protocol](docs/protocols/exploratory_training.md).
- Clinical annotation correctness and patient linkage remain unresolved. No glaucoma disc/cup, vascular/hypertension or DR-specific segmentation model has been trained in this project.
- Native C++ currently provides a tested weighted-mean interoperability example. Neural-network training is Python/PyTorch; custom CUDA kernels and C++ autograd are not implemented.

Training saves configuration, software version, source revision when available, working-tree state, archive/manifest hashes, optimizer-step counts, timing and memory. A wheel without Git metadata records the source revision as unavailable instead of borrowing the caller’s repository. CPU/GPU numerical equality is not guaranteed; preserve the environment and hardware record.

## Cross-task experiment programme

A [prespecified programme](docs/protocols/cross_task_experiments.md) covers OTFID, FIVES vessels, IDRiD lesions and REFUGE2 disc/cup: 36 core designs (including the existing nine OTFID runs) plus nine conditional objective ablations. Seven clean/corrupted evaluation conditions reuse each core checkpoint. These are plans, not additional trained capabilities; FIVES and IDRiD now have separate adapters and feasibility runners. REFUGE2 access and task-specific objective ablations remain pending.

```bash
PYTHONPATH=src python scripts/prepare_cross_task_experiments.py --output outputs/cross-task-design-v1
```

The generator writes a non-executable JSON/CSV register with missing metrics as null and preserves existing output directories. It never starts training.

## Data boundaries

| Dataset | Current or intended role | Constraint |
|---|---|---|
| OTFID v3 | Binary toxoplasmosis lesion pilot | No verified paired narrative EMR; patient identities unresolved |
| FairVLMed | Separate glaucoma image–text research | Original-note diagnosis and temporal leakage review required |
| BRSET | Fundus and structured clinical variables | Credentialed access; metadata are not complete narrative records |
| OLIVES / GRAPE | Compatible longitudinal ophthalmic tasks | Different cohorts and endpoints; no automatic time-to-blindness target |
| FIVES / RETA / PAPILA | Candidate vessel or disc/cup extensions | Separate annotations, access review and experiments required |

Never combine unrelated patients across datasets. Dataset licenses and access conditions are independent of the code license. Public code contains no patient images, notes or third-party model weights. See [dataset cards](docs/dataset_cards/README.md).

## Repository guide

| Directory | Responsibility |
|---|---|
| `src/ocuweave/data/` | Adapters, paired geometry, grouping and audit modules |
| `src/ocuweave/models/` | Encoder, query head, LQNet and architecture registry |
| `src/ocuweave/training/` | Training/checkpoints and controlled experiment runners |
| `src/ocuweave/evaluation/` | Metrics, original-grid restoration and diagnostics |
| `src/ocuweave/inference/` | Checkpoint-to-image prediction and artifact export |
| `native/` | Optional C++17 numerical interoperability example |
| `configs/`, `scripts/`, `notebooks/` | Research recipes, advanced entry points and Colab launchers |
| `tests/` | Geometry, metrics, gradients, CLI round trips and native parity |
| `docs/`, `paper/` | Protocols, results, model cards and manuscript source |

Clinical/language/fusion folders currently describe future interfaces. Their presence is not a completed multimodal model. See [API and extension boundaries](docs/api.md).

## Development

```bash
python -m unittest discover -s tests -v
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

Set `OCUWEAVE_NATIVE_LIBRARY` to the absolute shared-library path to include Python/native parity checks (`build/libocuweave_native.so` on Linux; `.dylib` on macOS). The [CI workflow](.github/workflows/ci.yml) checks native/Python behavior and runs the public CLI from an installed wheel outside the checkout.

Contributions should include a reproducible issue, a small scoped change and relevant checks. Describe the data license and provenance for any proposed dataset integration; do not submit patient data or checkpoints in a pull request. See [CONTRIBUTING](CONTRIBUTING.md) and [release procedure](docs/protocols/releases.md).

## FIVES vessel experiments

Prepare the official extracted FIVES release and run a supplementary feasibility comparison:

```bash
python scripts/run_fives.py prepare --root /path/to/fives --manifest /path/to/new-manifest.json
python scripts/run_fives.py pilot --root /path/to/fives --manifest /path/to/new-manifest.json --output /path/to/new-run --device cpu --steps 100
```

This compares Query 192, Query 384 and LQNet 384 using seed 42 and equal budgets. It saves final checkpoints, original-grid validation masks, Dice/IoU, centerline metrics and selected examples. It does not complete the separate 500-step GPU programme. Official test images are not evaluated. Patient identity remains unverified; read the [feasibility protocol](docs/protocols/fives_feasibility.md).

## IDRiD lesion experiments

Use the original segmentation subset, keeping the four lesion annotations separate:

```bash
python scripts/run_idrid.py prepare --root /path/to/idrid --manifest /path/to/new-idrid-manifest.json
python scripts/run_idrid.py pilot --root /path/to/idrid --manifest /path/to/new-idrid-manifest.json --output /path/to/new-idrid-run --device cpu --steps 100
```

The four heads predict microaneurysms, haemorrhages, hard exudates and soft exudates. Unavailable masks are excluded from supervision and evaluation. Read the [source and missing-annotation policy](docs/protocols/idrid_feasibility.md) before use. Both feasibility runners require the research dependencies and an extracted dataset; they do not download data implicitly.

Predict four lesion channels from a trained IDRiD checkpoint:

```bash
python scripts/predict_idrid.py --checkpoint /path/to/run/lqnet384/final.pt \
  --image /path/to/fundus.jpg --output /path/to/new-prediction --device cpu
```

This saves four original-size binary masks and a provenance receipt. Add `--save-probabilities` for the float probability array. Use this task-specific command for IDRiD; the generic `ocuweave predict` command is for binary checkpoints.

## Citation and license

Please cite the **software** until a paper is publicly available. GitHub’s “Cite this repository” uses [CITATION.cff](CITATION.cff).

```bibtex
@software{shevchenko2026ocuweave,
  author = {Shevchenko, Marchel Andrian},
  title = {OcuWeave: Evidence-Grounded Vision--Language Learning with
           Incomplete Clinical Records for Ophthalmic Disease Assessment},
  year = {2026},
  version = {0.10.0},
  url = {https://github.com/chenko-the-sorcerers/ocuweave}
}
```

Original code: [MIT](LICENSE). Research use only; the software is not validated for patient care. Version numbers track software changes, not diagnostic quality or clinical approval.
