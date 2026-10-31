# Cross-task exploratory programme v1

Marchel Andrian Shevchenko · 1 November 2026. This is a design, not a completed multi-disease model. Ocular toxoplasmosis remains the primary research target. All new cohorts, adapters, multi-output heads and additional objectives require implementation and audit before execution. No patient pairs are invented across datasets; EMR is excluded.

## Ordered execution and resource budget

1. Continue the existing nine-run OTFID suite using its immutable v0.8.1 source/data contract after CUDA smoke passes. Do not substitute a new package version into that suite.
2. Prepare FIVES first. Audit the official archive, paired filenames, mask values, geometry, quality, duplicate families and any available patient linkage. Keep the official test partition untouched; define development validation only within the official training partition. Published patient counts do not prove that our files provide usable patient IDs.
3. Run the three FIVES core variants at seed 42 after a full-batch GPU smoke check. Complete seeds 123 and 2026 under the same contract. Evaluate all runs, including failures; no early stopping of the programme based on favorable scores.
4. Implement and test IDRiD four-channel supervision and REFUGE2 nested disc/cup supervision. Audit each release and freeze separate development manifests before their seed-42 gates. A missing class-mask file is unknown supervision until the release documentation establishes that it means an empty mask.
5. Run conditional task-specific objectives only after their implementation, coefficients and evaluator are registered. If the GPU budget cannot cover the design, report the unexecuted rows; do not silently reduce steps/batch or hide failed seeds.

Generate a fresh register from the repository root:

```bash
PYTHONPATH=src python scripts/prepare_cross_task_experiments.py --output outputs/cross-task-design-v1
```

This writes JSON and CSV with 45 stable run IDs. It does not download data or execute training. The register is a snapshot as of 1 November 2026; it must not be treated as a live result tracker. There are 36 core designs (including the existing nine OTFID runs) and nine conditional designs. Thus 36 additional runs are proposed, not 45 additional runs.

## Core controlled comparisons

For each dataset, compare Query 192, Query 384 and LQNet 384 with seeds 42, 123 and 2026. Query 192 → Query 384 isolates input resolution; Query 384 → LQNet 384 tests the local-detail branch. The latter adds parameters and is not a capacity-matched control. All start from random weights without U-Net or pretrained backbones.

The new-task starting design uses batch 8, 500 optimizer steps, AdamW (learning rate 0.001, weight decay 0.0001), cosine step scheduling, gradient clipping at 1, float32, paired horizontal flips at probability 0.5, and masked BCE plus soft Dice averaged over supervised channels. Fix sample order, available-label policy and validation schedule across variants within each dataset. Task-specific channel counts necessarily differ across datasets; do not pool their scores as one benchmark.

Use RGB/255, aspect-preserving letterbox, bilinear image interpolation and nearest-neighbor mask interpolation; exclude padding from objectives. Audit masks and small structures at 192/384 **before** freezing the new-task configuration. If important annotations disappear, amend the resolution/tiling design before any performance-based selection. Tiling, when needed, requires image/patient grouping before crops, a fixed sampling policy, full-image reconstruction and its own controlled comparison. It is not silently added to the 192-versus-384 experiment.

New-task checkpoint selection is highest validation primary metric at fixed 50-step intervals, earliest checkpoint on ties. Restore probabilities to original dimensions before thresholding at 0.5. The existing OTFID suite retains its original 20-epoch, input-grid checkpoint-selection policy; document that distinction instead of retroactively changing it. No new-task contract is considered frozen until exact code/data/configuration hashes, channel policies, evaluator conventions and available compute are recorded.

## Anatomical targets and endpoints

| Task | Supervision | Primary endpoint | Secondary analysis |
| --- | --- | --- | --- |
| OTFID | Binary annotated lesion union | Positive-image macro Dice | IoU, small-union Dice, healthy FP area; existing protocol unchanged |
| FIVES | Binary vessel tree | Per-image vessel Dice | IoU, sensitivity, precision, clDice and thin-reference-centerline recall |
| IDRiD | MA, haemorrhage, hard/soft exudates in four independent channels | Mean per-class positive-image Dice over observed labels | Per-class IoU/PR-AUC, connected-component sensitivity and FP components per image |
| REFUGE2 | Whole disc (including cup) and cup in two channels | Mean of per-structure image-macro Dice | Per-structure IoU, boundary distance, vertical cup/disc ratio error, cup-outside-disc area |

Secondary evaluators are planned, not implemented claims. Define connectivity (8-neighbor), pixel calibration and empty-reference behavior explicitly. Thin-vessel strata use reference skeleton radius in original pixels; report radius bins (≤2, >2–4, >4) as image-resolution-dependent, not physical vessel caliber. For DR, match components one-to-one and report several prespecified IoU thresholds (0.1, 0.25, 0.5); report empty-case false positives separately and never infer an absent annotation to be healthy. Publish denominators per class, including undefined strata as null. Disc/cup ratio error is undefined for invalid/absent discs; count exclusions and do not substitute zero. The cup must not be excluded from the whole-disc reference when converting palette masks.

Synthetic tests must cover perfect/disjoint/empty masks, disconnected vessels, tiny lesions, overlapping class channels, invalid disc geometry, crop reconstruction and padding. Metrics measure anatomy; neither vessel masks nor disc/cup ratios alone establish disease diagnosis. No time-to-blindness target is available here.

## Conditional objective ablations: nine runs

Each adds one term to LQNet 384, using the same dataset, seed, optimizer budget and baseline predictions policy. The control is the matching core LQNet run, never a different resolution.

- FIVES: a soft centerline-consistency term, motivated by clDice [4]. Lock its implementation and coefficient before this comparison; do not label it a new invented loss.
- IDRiD: class-balanced loss using training-only prevalence, with a prespecified clipping rule. No validation/test prevalence enters weights. Missing annotations remain masked.
- REFUGE2: cup-containment penalty, mean of `p_cup * (1 - p_disc)` over valid pixels. This is a proposed consistency term, not evidence of anatomical validity. Fix the coefficient without inspecting test predictions.

The coefficients and class-weight clipping rule remain unresolved design parameters. These nine rows are explicitly conditional, not runnable configurations. Selecting a coefficient on development data requires a separately logged tuning budget; its runs cannot be hidden within the main comparison.

## Evaluation without retraining

For every completed core checkpoint, evaluate clean input and six fixed corruptions: Gaussian blur σ=1/2 original pixels, intensity multipliers 0.8/0.6, and additive RGB Gaussian noise σ=0.01/0.03 on [0,1] images (clip afterward). Apply corruption before resizing; leave annotations unchanged. Use deterministic per-image noise shared across variants and seeds. These synthetic conditions test controlled sensitivity, not a simulation validated against clinical image degradation.

Report paired degradation against that checkpoint's clean validation score, stratified by existing quality flags and source disease/device when labels are available. Do not train on test images, relabel poor-quality images, or present corrupted variants as new patients. This is seven evaluation conditions per checkpoint, not seven additional training runs. Cross-dataset external validation is a separate future extension requiring compatible label definitions and access; FIVES disease strata are not external datasets.

## Publication and result admission

Each completed run must have source commit, environment, manifest/archive/config/checkpoint hashes, full optimizer-step counts, original-grid metrics with class denominators, runtime/memory, failure status and completion receipts. A reported zero is valid only when measured. Missing results are null. Publish per-seed values, mean and sample SD, and paired resolution/detail differences; seed variability is not patient-level uncertainty or a significance test.

On the research page, separate historical pilot results, current frozen OTFID development, new-task plans, and completed task-specific tables. Keep reference masks visually distinct from predictions. Select gallery examples deterministically across low/median/high overlap and false-positive cases, state the selection rule, and retain failures. All image redistribution needs the dataset's applicable terms. No screenshots of raw notebook output should become the public research figure.

## Sources

[1] K. Jin et al., “FIVES: A Fundus Image Dataset for Artificial Intelligence based Vessel Segmentation,” Scientific Data, 2022, doi:10.1038/s41597-022-01564-3.

[2] P. Porwal et al., “Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Diabetic Retinopathy Screening Research,” Data, vol. 3, no. 3, Art. 25, 2018, doi:10.3390/data3030025. Original dataset-definition exception to the recent-literature window. [Official data description](https://idrid.grand-challenge.org/Data/).

[3] H. Fang et al., “REFUGE2 Challenge: A Treasure Trove for Multi-Dimension Analysis and Evaluation in Glaucoma Screening,” arXiv:2202.08994, 2022. [Source](https://arxiv.org/abs/2202.08994).

[4] S. Shit et al., “clDice—A novel topology-preserving loss function for tubular structure segmentation,” CVPR, 2021. [Official proceedings](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html).
