# Executed annotation/preprocessing review and experiment readiness

Measured 2026-10-31 using v0.4.0, commit
`a2aca48e3225a220aef8f46b54987e9ef75787c4`. Documentation recorded in v0.4.1.
[Executed notebook](https://colab.research.google.com/drive/1aT_KxLrJ5ymEy1JAoOg9rin2tG0N9nGK).

## Scope and completed checks

All **238 development pairs** (196 training, 42 validation) matched the frozen
image dimensions, mask dimensions and decoded foreground counts. There were zero
pair failures. Test images were not loaded. There were 126 positive and 112 healthy
images. Six deliberately adverse validation cases received original/192/384
reference-boundary overlays. These are geometric checks and visual inspection,
not ophthalmologist adjudication or proof of image–patient identity.

The eight protocol code cells completed on CPU without errors. Step 6 registered
the nine planned experiments and skipped training because CUDA was unavailable;
completion of the notebook does not mean the training experiments ran. A subsequent
T4 allocation attempt was rejected by Colab due to GPU usage limits. No paid
compute was purchased. A supplementary cell stores the remaining review panels.

## Measured resampling effects

Masks are letterboxed with nearest-neighbor interpolation and restored to their
original dimensions with nearest-neighbor interpolation. Metrics compare this
roundtrip mask to the unchanged original mask. They do not involve model predictions.
“Small” means >0–1% foreground area relative to the original full image.

| Positive-image group | n | Mean roundtrip Dice, 192 | Mean roundtrip Dice, 384 |
|---|---:|---:|---:|
| All development | 126 | 0.9767266144 | 0.9879767566 |
| Small lesions | 13 | 0.9250862722 | 0.9618751623 |
| Training | 104 | 0.9777477106 | 0.9886245527 |
| Validation | 22 | 0.9718996142 | 0.9849144479 |

No positive mask disappeared entirely at either resolution. This does not imply
all components or fine boundaries survived.

| Positive-image group | n | Exact boundary Dice, 192 | Exact boundary Dice, 384 | Mean absolute relative area error, 192 | Mean absolute relative area error, 384 |
|---|---:|---:|---:|---:|---:|
| All development | 126 | 0.1386230618 | 0.2543922399 | 0.0039166292 | 0.0017900700 |
| Small lesions | 13 | 0.1322402800 | 0.2428992957 | 0.0145379424 | 0.0073247537 |
| Training | 104 | 0.1403387546 | 0.2574112361 | 0.0035841502 | 0.0017197459 |
| Validation | 22 | 0.1305125141 | 0.2401206215 | 0.0054883477 | 0.0021225111 |

Exact boundary Dice uses a one-pixel, four-neighbor boundary on the original grid,
with **zero tolerance**. It is highly sensitive to one-pixel shifts and is not a
clinically validated boundary score. Relative area error is a fraction, not a
percentage in this table. Both statistics are descriptive preprocessing measures.

The overlays inspected show reference contours at the same image geometry, with
no obvious global displacement in the selected cases. This does not establish
clinical contour accuracy. Original mask threshold rules remain unchanged.
No manual cleaning, correction, deletion or relabeling was performed.

## Interpretation

The 384-pixel roundtrip preserves reference-mask geometry better on average,
including small lesions. This supports testing higher input resolution; it does
not predict a model-performance gain. Total mask disappearance does not explain
the earlier small-lesion failures. Image detail, label quality, optimization and
model representation remain possible contributors; this review does not isolate
their effects. No segmentation improvement is claimed without completed runs.

## Current experiment status

| Task | Verified status |
|---|---|
| Technical image–mask/resize review | Completed, clinical boundary review incomplete |
| Query 192 vs query 384 | Implemented and registered; not trained |
| Query 384 vs local-detail query 384 | Implemented and synthetic-tested; not trained |
| Seeds 42, 123, 2026 across three variants | Nine runs pending, zero completed; 500 updates/run planned |
| Common original-grid evaluation, memory/time logging | Implemented and synthetic integration-tested; no real-run measurements yet |
| FairVLMed original-EMR multimodal study | Separate glaucoma protocol; semantic/temporal review not complete; not trained |

The original test-result JSON files retain their previously documented SHA-256
values. All new model comparisons use development data only. A new independent
held-out evaluation and verified patient linkage remain necessary for stronger
claims. No toxoplasmosis image is paired with another cohort's clinical note.

## Traceable artifacts

Under the authorized Drive project:

- `02 — Audits/development-v0.4.0/annotation-geometry/summary.json`: overall checks and provenance.
- `per_image.json` in that directory: every pair at both resolutions, area, boundary and roundtrip values, and intermediate mask-value counts.
- `descriptive-strata.json`: exact values in the tables above, computed from saved rows and frozen image dimensions.
- `overlay-00.png` through `overlay-05.png`: six reference-geometry panels; selection is intentionally adverse, not representative.
- `02 — Audits/development-v0.4.0/execution-status.json`: GPU limitation and uncompleted clinical/EMR gates.
- `04 — Runs/v0.4.0/resolution-detail-seeds/`: frozen source/config/input contract, nine pending entries in CSV/JSON/Markdown. No new checkpoints exist for these runs.
- `05 — Research documentation/v0.4.0/20261031T054208Z/`: initial documentation snapshot with 129 indexed artifacts and consolidated source results. It is a point-in-time inventory, not a claim that subsequently edited notebook bytes are unchanged.
- Final v0.4.1 documentation snapshot: refreshed results/protocols and inventory after final saved notebooks; earlier snapshots remain preserved.

Software validation: 32 Python tests including C++ parity; native CMake/CTest,
synthetic audit, wheel packaging, notebook schema/code syntax, and GitHub CI passed.
