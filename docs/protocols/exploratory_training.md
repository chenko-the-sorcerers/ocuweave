# Exploratory development training v1

Author: Marchel Andrian Shevchenko · 2026. Software v0.6.0. No U-Net, pretrained weights or EMR.

## Cohort revision

Use the immutable v0.2.0 eligible manifest and the completed v0.5.0 near-duplicate/quality audit. Compute transitive components over existing duplicate, similarity and any supplied patient links, plus the recorded development candidate pair. A candidate-connected development component containing a training image goes entirely to training; a validation-only component remains validation. This is a conservative, deterministic choice made without model performance. It does not verify a duplicate or patient identity. Any test-involving bridge fails closed and requires a separate protocol.

Write a new manifest, SHA-256 receipt, counts and per-record decision log in private Drive. Never modify the old manifest, test rows, results, checkpoints or source archive. Retain all technical quality flags and all 238 development images; no clinical clearance or mask correction is implied. Do not silently reuse historical split-count summaries: the new decision file is controlling.

## Fixed comparisons

In order: seed 42 (Query 192, Query 384, LQNet 384), then seeds 123 and 2026 in the same variant order. Width 24, feature dimension 48, eight queries, batch 8, 20 epochs, learning rate 0.001, AdamW weight decay 0.0001, cosine epoch schedule, gradient clipping 1, float32. Every run has the same sample membership, paired horizontal flip probability 0.5, letterboxing, valid-padding mask, BCE + soft Dice and epoch/batch schedule. Configs differ only in resolution, local-detail architecture, or seed. There is no early stopping or adaptive batch reduction. The locked protocol calculates and stores the exact shared optimizer-step budget from the revised training size before execution.

DataLoader ordering and augmentation use the same seed across paired variants. Each run starts from random initialization; the local residual projection starts at zero, and its shared base initialization matches the query implementation. LQNet adds parameters; this is not a parameter-matched control. GPU kernels may still have platform-dependent numerical variation; record torch, device and source versions.

## Pipeline gates

Before research training, run three full training batches for each seed-42 variant. Check image/mask/valid geometry and range, finite losses and every gradient, nonzero global gradient norm, checkpoint save/reload equality, time and allocated/reserved memory. Save paired reference-mask overlays to private Drive for visual inspection. This is technical alignment checking, not clinical annotation validation. A CPU check is distinctly labelled and cannot unlock GPU training. Smoke checkpoints never initialize research models.

A passed CUDA smoke receipt must match the frozen manifest, archive, source commit and pipeline contract. Run directories are immutable once completed. A partial run stops the suite rather than silently being counted or restarted; completed artifacts are hash-verified before reuse. If an interruption requires a new attempt, preserve the partial directory and document it separately.

## Validation endpoints

Select each checkpoint by validation positive-image Dice on its own input grid, a fixed choice retained from the preceding protocol. Restore probabilities to original image dimensions before thresholding at 0.5 for the cross-resolution comparison. Report positive-image macro Dice and IoU, all-image macro IoU, small-lesion Dice and count (positive reference union area ≤1% of the entire original image), and mean healthy false-positive area fraction. This small-image-area stratum is not a connected-component lesion sensitivity measure. Include any-pixel healthy false-positive rate, time, optimizer steps, parameters, peak CUDA allocated/reserved memory and process RSS. CPU RSS is process lifetime; run time includes validation/checkpoint IO and excludes setup/final evaluation. Validation decode/evaluation time is separate.

Three-seed means, sample standard deviations and paired differences are descriptive initialization variability; image-bootstrap intervals do not establish patient independence. The already-observed historical test is never used for change selection or rerun here. No new independent-test or clinical-validity claim is supported. FairVLMed glaucoma and its original EMR remain outside this pipeline until their separate semantic/temporal review is complete.
