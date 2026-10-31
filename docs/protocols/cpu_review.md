# CPU photo-mask review and evaluator verification

This review does not train models, modify labels or change splits. Use the verified
historical manifest for continuity; repeat on the new development manifest after it
is frozen and retain both output directories. Test image payloads are never decoded.

```sh
python scripts/build_review_bundle.py --manifest PATH/manifest.json --archive PATH/source.zip --output PATH/new-review --allow-unverified-patients
python -m unittest tests.models.test_metric_oracles tests.data.test_review_bundle -v
python scripts/prepare_run_report.py --output PATH/new-report-template
```

The explicit unverified-patient flag records the limitation, not clinical approval.

## Reviewer sheet

`image-mask-review.csv` has one row per development image, including failed technical
pairs. Fields: visual alignment, boundary completeness, blur, illumination, artifacts,
retina cutoff and clinical lesion correctness. Allowed human review values are
`unreviewed`, `acceptable`, `concern`, `indeterminate`. Record reviewer, role, date,
evidence and rationale. Do not mark clinical correctness based on a dimension check.
Default disposition is `retain_pending_review`; no automatic deletion is supported.

Quality proxies are fitted on training images only. Low Laplacian variation is a
sharpness proxy; mean intensity, clipping and quadrant differences are illumination
and artifact proxies. Edge contact of non-dark support is NOT a retinal segmentation
or proof that anatomy is truncated. Human review must distinguish these explanations.

Overlays show unchanged RGB and mask, the 192/384 input grids, and original-grid
reference boundaries (green original, magenta roundtrip). They do not show predictions.
Eight deterministic examples prioritize flags, small positive unions and controls;
this subset is not a full-cohort visual or clinical review.

## Resize measurements

For each image and size, record content dimensions, all four padding widths, padding
fraction and foreground-in-padding check, original/resized foreground counts and
area fractions, roundtrip Dice/IoU, exact-boundary Dice and signed relative area error.
Empty-target relative error is missing, not zero. Exact-boundary Dice has zero tolerance
and should not be presented as a clinical boundary score.

Four-connected components are labeled on the original thresholded mask. Their labels
are sampled with the same nearest-neighbor transform as the binary target. A component
is lost if no output content pixel samples its label. Counts may reflect JPEG mask
artifacts and do not represent validated lesion instances. Pixel areas are not mm².

## Evaluator oracles

Tests use independently specified TP/FP/FN and interpolation values: perfect, partial,
disjoint, empty/empty, false-positive-only and false-negative-only masks; padding
exclusion; positive-only macro aggregation; rectangular padding removal; bilinear
restoration of [0,1] to [0,.25,.75,1]; threshold equality at 0.5; invalid/NaN inputs.
Nearest-neighbor reference restoration and bilinear probability restoration remain
distinct operations. A test pass establishes implementation behavior, not clinical validity.

## Nine-run report

The report has one row per seed/variant before any training. Missing measured values
remain blank in CSV and null in JSON. It includes full configuration, source commit,
manifest/archive/checkpoint SHA-256, Dice/IoU, small-union count and Dice, healthy FP area,
training and evaluation seconds, parameter/step counts, device and memory measures.
A template without data hashes is not a registered experiment. Use the existing
exploratory registration and CUDA smoke gate before training. Per-image artifacts
remain private; only aggregate, provenance-checked findings belong in public docs.
