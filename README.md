# OcuWeave

**Evidence-Grounded Vision–Language Learning with Incomplete Clinical Records for Ophthalmic Disease Assessment**

An open research project by **Data Sorcerers**, investigating how retinal images,
clinical notes, and structured clinical information can be combined when evidence
is incomplete or inconsistent. Ocular toxoplasmosis is a focus of the research agenda.

## Current status

**Pre-experimental research scaffold.** This release provides a patient-level data
contract, synthetic examples, manifest auditing, experiment specifications, and a
tested C++/Python interoperability example. It does **not** contain a trained
OcuWeave model, validated diagnostic predictions, benchmark results, or a published
paper. The model, objectives, and training protocol remain research proposals.

## Quick start

Python 3.10+; CMake 3.20+ and a C++17 compiler for the optional native example.

```sh
python -m venv .venv
# Activate the virtual environment for your operating system.
python -m pip install -e .
python scripts/audit_data.py examples/synthetic_manifest.json
python scripts/prepare_data.py examples/synthetic_manifest.json --output outputs/audited_manifest.json
python -m unittest discover -s tests -v
```

The example manifest contains fabricated identifiers and placeholder image paths;
it contains no patient data. For real data, use `audit_data.py --check-files` with
`--data-root` pointing to the authorized local dataset directory.

```sh
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure
python scripts/benchmark_ops.py --library build/libocuweave_native.dylib
```

On Linux use `build/libocuweave_native.so`; on Windows locate
`ocuweave_native.dll` in the build output. For Python/native parity tests, set
`OCUWEAVE_NATIVE_LIBRARY` to the absolute library path before running unittest.
The CPU operator is a numerical interoperability example, not the proposed fusion
algorithm. It has no autograd integration and must not be used as a differentiable
training operator. CUDA kernels are not implemented yet.

`train.py`, `evaluate.py`, and `predict.py` intentionally return an explicit
not-implemented error after parsing arguments. They never fabricate weights,
predictions, or metrics. Their `--help` describes the planned entry points.

## Research direction

- Visual representations for full retinal images and lesion regions.
- Clinical representations that preserve text evidence and missingness.
- A proposed evidence-alignment and reliability-aware fusion method.
- Evaluation against image-only, text-only, tabular, and late-fusion baselines.
- Separate evaluation of prediction, localization, calibration, and robustness.

See [architecture](docs/architecture.md), [evaluation protocol](docs/protocols/evaluation.md),
[dataset cards](docs/dataset_cards/README.md), and [model card](docs/model_cards/ocuweave.md).

## Dataset plan

| Dataset | Intended role | Important boundary |
|---|---|---|
| Harvard-FairVLMed | Paired SLO images and original clinical notes | Glaucoma; notes can contain the target diagnosis. Audit leakage first. |
| BRSET | Color fundus + limited EMR-derived metadata | Scar (toxoplasmosis) labels do not establish active infection. Credentialed access. |
| OTFID v3 | Toxoplasmosis lesion classification/segmentation | No paired narrative EMR verified; patient grouping requires inspection. |
| OLIVES | OCT/near-IR imaging and longitudinal clinical measurements | Not a full narrative EMR or a toxoplasmosis cohort. |
| mBRSET | External evaluation on compatible DR endpoints | Different camera/cohort; not an active-toxoplasmosis benchmark. |

Never pair an image with another dataset's patient record. Text generated from
metadata is synthetic/serialized metadata, not an original clinical note.

## Repository map

- `configs/`: dataset, model, experiment, and ablation TOML specifications.
- `src/ocuweave/`: Python data/model contracts, reference operators, and audit logic.
- `native/`: C++17 operator, C ABI binding, and optional future CUDA kernels.
- `scripts/`: explicit research entry points.
- `tests/`: data integrity, model contracts, and native/reference parity.
- `paper/`: manuscript, figures, and tables for future research outputs.
- `data/`, `outputs/`: ignored local data and generated artifacts.

The directory READMEs describe unimplemented modules without pretending that the
method already exists. Experiments must record patient splits, seed, code revision,
configuration, data provenance, and checkpoint origin.

## Public release and licensing

Original code in this repository is MIT licensed. Dataset and pretrained-weight
licenses remain separate. No patient images, clinical notes, or third-party weights
are redistributed. Public checkpoints are a goal, conditional on upstream terms
and appropriate privacy review. This scaffold is for research, not patient care.

## Reference implementations

- [FairCLIP (CVPR 2024)](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP)
- [RETFound](https://github.com/rmaphoh/RETFound)
- [EyeCLIP](https://github.com/Michi-3000/EyeCLIP)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)

These are research references; no upstream implementation or checkpoint is vendored.
See `CITATION.cff` to cite this software scaffold; no publication acceptance is claimed.
