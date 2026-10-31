# Annotation, resolution and local-detail development protocol

Author: Marchel Andrian Shevchenko. Registered 2026-10-31. Software v0.4.0.
Implementation and registration do not constitute executed results.

## Questions and order

1. Are retained development image/mask dimensions and frozen foreground counts
consistent? How much geometry changes in a 192/384 nearest-neighbor roundtrip?
2. Does changing input size alone improve validation lesion segmentation?
3. At the same input size, does a local residual detail branch add value?
4. Are observed effects consistent across the three predeclared seeds?

Review all 238 training/validation pairs. Test images and test-note content are
not loaded by the new review or development evaluator. Preserve all raw data,
annotations, the frozen binary-v2 split and the previously saved test results.
A dimension/foreground mismatch blocks training. Clinical boundary adjudication
remains separate and incomplete; a technical pass cannot replace an ophthalmologist.

The previously selected adverse validation examples receive original-image
contour panels: original mask, 192 roundtrip, 384 roundtrip. No inference mask is
shown in this geometry panel. Quantify positive-mask roundtrip Dice, exact-grid
four-neighbor boundary Dice, relative area change, disappearance, and intermediate
JPEG mask values. These are transformation diagnostics, not model scores.
Do not use the original mask to select image crops at inference.

## Predeclared matrix

| Variant | Architecture | Input | Seeds | Purpose |
|---|---|---:|---|---|
| query192 | Existing query segmenter | 192 | 42, 123, 2026 | Contemporaneous control |
| query384 | Same query segmenter | 384 | 42, 123, 2026 | Resolution-only contrast |
| detail384 | Query + local residual branch | 384 | 42, 123, 2026 | Detail-only contrast |

Run each seed's three variants in that order. Each run starts from random weights;
none reuses the pilot checkpoint. Same seed initializes the shared query modules
identically, and a separately seeded loader controls sample order. Extra detail
parameters make the architecture comparison not parameter-matched.

Keep 20 epochs × ceil(196/8) = **500 optimizer steps**, batch 8 (last batch 4),
width 24, dim 48, eight queries, AdamW lr .001 and weight decay .0001, cosine decay,
gradient clipping 1, BCE + Dice, horizontal training flips, and the same split.
No early stopping or automatic OOM batch-size changes. The same update budget does
not imply equal FLOPs or wall time. Log hardware, library versions and compute.
If interruption/OOM occurs, preserve the folder and logs; incomplete runs are not
scored and cannot silently resume. A changed protocol requires a new suite folder.

## Experimental architecture

The original stride 4/8/16 compact encoder and query mask head remain unchanged.
The local branch applies a 3×3 stride-2 RGB convolution, GroupNorm/GELU, a 3×3
 depthwise convolution and GroupNorm/GELU. Concatenate the half-resolution local
features with resized coarse logits; use 1×1 mixing/GELU and a 1×1 residual-logit
projection, then bilinearly restore/add the correction to the coarse output.
The final projection begins at zero, so initial predictions exactly equal the
shared base model; gradients reach earlier detail layers after the projection
starts learning. There are no pretrained weights or U-Net modules.

This simple residual branch is an engineering hypothesis. It is not asserted to
be a novel published contribution, evidence-grounded EMR fusion, or a proven fix.
False-positive control is measured here; a changed loss or gating objective would
be a separate future ablation, not folded into this comparison.

## Selection, metrics and resources

Select each checkpoint by validation positive-image Dice on its input grid,
unchanged from the pilot rule. For **cross-resolution comparisons**, restore
probabilities (crop padding, bilinear resize, threshold .5) to the **same original
image grid** and compare against the unchanged original reference masks.
This common evaluation grid avoids changing the target discretization between
variants. Selection remains input-grid dependent and is explicitly recorded;
a future common-grid selection experiment would require a separate protocol.

Record original-grid positive Dice, global Dice, IoU, source/label/area strata,
small-lesion Dice (>0–1% original full image area), healthy any-pixel FP and mean
healthy FP area/full image. Preserve per-image counts. Checkpoints, JSON, CSV,
Markdown, and figures stay in Drive. The old 192-grid pilot scores are historical
context, not the control of the new original-grid experiment.

Run each training process separately. Record parameters, device/model, Torch and
Python versions, wall seconds for the epoch loop including validation/checkpoint
I/O, completed/planned optimizer steps, peak process RSS (lifetime high-water mark),
and CUDA peak allocated/reserved bytes since reset before model creation. CPU CUDA
measurements are null. Final evaluation time includes decoding and is reported
separately. Different hardware comparisons must be qualified.

Report every seed, mean and sample SD where at least two values exist, plus paired
seed differences for query384-query192 and detail384-query384. Three seeds support
descriptive stability checks, not strong significance claims. Image-bootstrap
intervals do not address unknown patient correlation. No test tuning, repeated
test scoring, clinical prognostic claim, or fabricated missing result is allowed.

## Documentation and interruption

`protocol.json` freezes source/config/input hashes; `runs.csv`, `summary.json` and
`README.md` show every planned run, including pending/incomplete runs. A completion
receipt hashes history, provenance, checkpoint and original-grid results. A separate
research-documentation snapshot inventories source reports with SHA-256 hashes.
The Colab notebook executes CPU review/tests first and runs training only when a
CUDA GPU is available. GPU quota failure is a recorded execution limitation, not
a result. No paid compute is purchased automatically.
