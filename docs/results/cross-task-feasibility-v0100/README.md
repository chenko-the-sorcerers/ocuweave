# Vessel and lesion feasibility experiments

Marchel Andrian Shevchenko · Data Sorcerers · 2026-11-01

Six measured, single-seed CPU runs are separate from the uncompleted 500-step core programme. Each uses random initialization, seed 42, 100 steps, batch eight, masked BCE plus soft Dice, AdamW (learning rate 0.001, decay 0.0001), cosine scheduling and paired horizontal flips. The fixed final checkpoint is evaluated once on validation; no test inference or validation threshold tuning is performed.

## FIVES vessels

| Variant | All-image Dice | Positive-image Dice | IoU | clDice | Thin-centerline recall | FP area | Training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| query192 | 0.1610 | 0.1540 | 0.0925 | 0.1405 | 0.0496 | 1.840% | 245.1 s |
| query384 | 0.1985 | 0.2002 | 0.1127 | 0.1762 | 0.0550 | 1.731% | 490.9 s |
| lqnet384 | 0.2022 | 0.2039 | 0.1150 | 0.1792 | 0.0540 | 1.661% | 383.4 s |

480 training / 120 validation / 200 preserved test images. Of the validation references, 119 are positive and one is empty. The empty reference (`train-447_G`) accompanies a visibly very dim/blurred photograph; it is retained for review and is not assumed healthy. Empty/empty overlap is defined as one. The two provisional similarity groups crossing official partitions bar official-test performance claims. Patient identities remain unverified. Vessels are anatomical targets, not hypertension labels.

## IDRiD lesions

| Variant | MA Dice | HE Dice | EX Dice | SE Dice | Mean class Dice | Training time |
|---|---:|---:|---:|---:|---:|---:|
| query192 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 135.3 s |
| query384 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 356.3 s |
| lqnet384 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 418.2 s |

43 training / 11 validation / 27 preserved test images. Four independent sigmoid heads predict microaneurysms (MA), haemorrhages (HE), hard exudates (EX) and soft exudates (SE). Missing annotation files are excluded from both loss and class metrics, not treated as negative labels. The native TIFF labels are decoded independently of their display palette; one opaque red/black RGBA mask has a separate validated decoder.

| Class | Annotated validation images | Positive validation images |
|---|---:|---:|
| MA | 11 | 11 |
| HE | 11 | 11 |
| EX | 11 | 11 |
| SE | 5 | 5 |

## Observed failure and next checks

All three IDRiD variants predict background on every annotated validation channel at the fixed 0.5 threshold: TP = 0 and FP = 0, with positive reference pixels remaining as false negatives. Dice and IoU are therefore exactly zero, not placeholders. This pilot has not learned usable lesion segmentation. Severe foreground imbalance, tiny-lesion downsampling and the short training budget are hypotheses to investigate; these runs do not isolate their effects.

For FIVES, local detail increases all-image Dice by 0.0037 over Query 384, but thin-centerline recall decreases from 0.0550 to 0.0540. There is no demonstrated thin-vessel preservation benefit. Before expanding runs, test training-only small-batch overfitting, inspect probability distributions, and register a patch/high-resolution comparison that changes sampling separately from the loss. Do not tune the held-out test or relabel missing annotations as background.

## Interpretation and limits

These are short feasibility experiments, not a robust ranking or clinical validation. Report zero overlap as an observed failure, not as missing data. Image-level bootstrap intervals in raw receipts do not establish patient independence. Full-image downsampling substantially reduces tiny-lesion representation: in reviewed IDRiD_01, the MA mask has 6,941 original positive pixels, 15 at 192 px and 54 at 384 px. Those counts describe two-dimensional sampling, not clinical lesion size. A separate high-resolution/patch protocol is warranted before stronger conclusions.

CPU training times include lazy decoding and some contention with acquisition, auditing and artifact export; they are not controlled throughput benchmarks. Memory is the process-lifetime RSS high-water mark, which can include preceding variants. GPU memory is unmeasured on CPU. Each completion receipt records its own checkpoint hash, source/configuration and validation metrics. The FIVES smoke initialization is discarded before training; IDRiD checks finite losses and gradients during training. Both verify checkpoint reload equivalence.

## Provenance

FIVES: [official Figshare v1](https://doi.org/10.6084/m9.figshare.19688169.v1), Jin et al., CC BY 4.0. Archive MD5 `789c80dd5376a82063e27fa49192bac9`, 1,764,974,308 bytes. IDRiD: Porwal et al., original segmentation subset redistributed through [Zenodo 17219542](https://zenodo.org/records/17219542), CC BY 4.0. Transfer MD5 `bdf5cd10b8b14d62e1178dda245cff60`, 584,315,841 bytes. This verifies the mirror transfer, not equivalence to an authenticated IEEE download. REFUGE2 remains unacquired because official access is unavailable.

- **FIVES** source `28db9c803802d3d8438f583d45d89e70b1d891ac`; manifest SHA-256 `632b76bad18375cb824fc18761216a00428310fe806f4170c2f866199f13ad68`.
- **IDRID** source `7e5accc2d2f3cd5db3d79cbeb0fe440cba58772c`; manifest SHA-256 `5efcc29e06b6bcb0a4d0a8cfd5a3600951485cbbef17f78ab10e8ed41d66a4d1`.

The JSON subdirectories contain completion receipts, per-image metrics, training histories, protocol settings and selected-example provenance. Full archives, checkpoints, audits and prediction masks remain in the authorized Drive project. Selected CC BY display figures are published on the [research page](https://data-sorcerers.com/ocuweave/#cross-task-results), with attribution and overlay transformations stated. The raw metric masks are never replaced by resized display images.

Protocols: [FIVES](../../protocols/fives_feasibility.md) · [IDRiD](../../protocols/idrid_feasibility.md).

## Recorded compute

| Dataset / variant | Parameters | Peak process RSS (GiB) | GPU memory |
|---|---:|---:|---|
| fives / query192 | 109,754 | 0.860 | Not measured (CPU) |
| fives / query384 | 109,754 | 1.809 | Not measured (CPU) |
| fives / lqnet384 | 111,411 | 2.098 | Not measured (CPU) |
| idrid / query192 | 192,416 | 0.909 | Not measured (CPU) |
| idrid / query384 | 192,416 | 1.980 | Not measured (CPU) |
| idrid / lqnet384 | 194,220 | 2.823 | Not measured (CPU) |

RSS is the process-lifetime high-water mark and may include preceding variants. It is not an isolated peak-allocation benchmark. The environment receipt records Python, PyTorch, image libraries, CPU and thread count.
