# OTFID v3 image-only pilot

Software: v0.2.0. Author: Marchel Andrian Shevchenko. Date: 2026-10-31.

## Question and scope
Can the compact query segmenter learn binary lesion masks, and how does it compare
with an FCN head using the same encoder and pixel decoder? This is a feasibility
pilot, not confirmation of novelty or clinical usefulness. Neither model uses
U-Net or pretrained parameters. The proposed EMR branch is not trained here.

## Source and cleaning
Use only the official OTFID v3 archive at https://zenodo.org/records/13883485
(CC BY 4.0). Preserve original bytes, official metadata, size, MD5 and SHA-256.
Decode images and masks; exclude missing/contradictory annotations instead of
inventing negative masks. Union active/inactive masks for this binary endpoint.
Keep every exclusion and mask-threshold warning in the audit JSON.

Remove exact decoded-pixel duplicates with identical masks. Exclude all conflicting
mask duplicates. Group retained images with dHash Hamming distance <= 4 and RGB thumbnail RMSE <= 0.025 conservatively
before splitting; candidates are not deleted. Inspect these candidates because
retinal appearance can produce false matches, and dHash can miss transformed duplicates.

Patient IDs must be supplied from verified records; filenames are never patient IDs.
When absent, the explicit exploratory flag permits a stratified image/group split
(approximately 70/15/15). Related patient, duplicate and similarity groups cannot
cross splits. Unknown cross-view patient overlap remains a limitation.

## Frozen pilot configuration
- Seed 42, 20 epochs, 192 x 192 letterbox, batch size 8, horizontal flips on training only.
- Compact encoder width 24; pixel embedding 48; 8 queries for the query model.
- AdamW, initial learning rate 0.001, weight decay 0.0001, cosine schedule, gradient clipping 1.
- BCE plus soft Dice; ignore letterbox padding. Raw pixel scaling to [0,1]; no dataset-fitted normalization.
- Select only by validation positive-image mean Dice, falling back to all-image Dice
  only if validation has no positive masks. Record that limitation if it occurs.
- Two architectures share data, encoder, pixel-decoder design and epoch budget.
  Query parameters are additional; this is not a parameter-matched comparison.

## Evaluation and interpretation
Fixed threshold 0.5; report positive-image and all-image macro Dice, macro IoU,
global Dice, and the fraction of healthy images with any false-positive pixel.
Empty-empty Dice is 1, so all-image Dice alone is insufficient. Image-bootstrap
intervals (500 resamples) do not correct for unknown patient correlation.
Metrics are measured at the configured letterboxed resolution. Original-resolution
inference restores probabilities geometrically but cannot recover lost detail.
Inspect small lesions, boundaries and acquisition artifacts before further claims.

The held-out test is evaluated after validation selection, once per frozen run.
Do not retune on these test results. Repeated seeds, patient linkage, external cohorts,
lesion-level and boundary metrics, and expert annotation review are subsequent work.

## Storage and interruption
Colab runs keep the original archive, manifests, environment, logs, best checkpoint,
SHA-256, configuration, source commit and metric JSON in Drive. Each epoch saves
history and any improved checkpoint. Completed runs are preserved. An interrupted
run requires a new run directory; optimizer/scheduler resume is not implemented.
The notebook can rerun safely after a completed run but must not silently reuse a
checkpoint from a different source revision, config, or manifest.

## Separate image–text cohort
FairVLMed acquisition is available through `scripts/download_data.py --dataset fairvlmed`.
It records the Hugging Face commit and file SHA-256 values.
`audit_fairvlmed.py` inspects filename pairing, official split counts, missing notes
and exact repeated notes without printing clinical text or decoding NPZ object arrays.
This is a structural audit, not completed image cleaning or a clinical leakage review.
The source uses CC BY-NC-ND 4.0; keep its access and redistribution terms separate
from the MIT software license. Never join its records to OTFID patients.
