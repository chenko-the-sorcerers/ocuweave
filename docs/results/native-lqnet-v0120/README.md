# Native LQNet: controlled baseline and weighted-loss diagnostic

**Snapshot: 1 November 2026, 06:08 UTC.** This is a training-only engineering study on four frozen IDRiD images. It is not validation, independent testing, an early-detection study, or a comparison with the historical PyTorch initialization.

The C++17 engine implements the tensor operators, analytic backward, LQNet, observed-label BCE/soft Dice and AdamW. Custom CUDA kernels run convolution/matrix products on a Colab T4; other operations run on the CPU. No ML framework, cuDNN or cuBLAS is linked. The frozen numerical source is [`0a4385c`](https://github.com/chenko-the-sorcerers/ocuweave/tree/0a4385cc354de17ff046e48a78395d28b3b1dd54), distinct from subsequent documentation/tests/checkpoint-I/O updates in the release.

## Measured baseline

384 px, seed 42, width 24, embedding 48, eight queries, random initialization, 200 optimizer steps, no augmentation, unchanged BCE + soft Dice, threshold 0.5. Metrics are computed after restoring probabilities to original image dimensions and then thresholding.

| Target | Positively annotated images | Macro Dice | Macro IoU | Pooled precision | Pooled recall |
|---|---:|---:|---:|---:|---:|
| Microaneurysms (MA) | 4 | 0.00% | 0.00% | Undefined: no predicted positives | 0.00% |
| Haemorrhages (HE) | 4 | 46.87% | 35.45% | 74.55% | 39.96% |
| Hard exudates (EX) | 4 | 42.58% | 28.79% | 63.63% | 46.15% |
| Soft exudates (SE) | 1 | 0.00% | 0.00% | Undefined: no predicted positives | 0.00% |

These are different aggregations: pooled precision/recall must not be inserted into a formula to reconstruct image-macro Dice. The SE sample count is one because missing annotations are unavailable, not negative examples.

The baseline has 194,220 parameters. Its measured training/final-inference/checkpoint-reload interval is 732.37 seconds. The highest sampled process GPU usage is 156 MiB, and host RSS is 1,386,432 KiB; these are sampled values rather than allocator peaks. The mixed CPU/CUDA reference backend prioritizes inspectable implementation and numerical checks, not optimized throughput.

The native baseline still fails the small-set fitting goal. MA/SE remain all-background at the fixed threshold. The measured result does not justify moving directly to a large research training run or claiming higher clinical accuracy. The weighted objective, longer budget, local-detail change and spatial router are separate hypotheses, not proven improvements.

## Paired weighted-objective result

The 200-step weighted run uses the same initial native model/optimizer checkpoint (verified identical SHA-256), input batch, architecture, seed, optimizer and step count. Only the BCE positive weights change; soft Dice is unchanged. Its measured interval is 735.37 seconds.

| Target | Baseline Dice | Weighted Dice | Weighted IoU | Weighted pooled precision | Weighted pooled recall |
|---|---:|---:|---:|---:|---:|
| MA | 0.00% | 0.00% | 0.00% | Undefined | 0.00% |
| HE | 46.87% | 54.01% | 40.62% | 45.88% | 61.72% |
| EX | 42.58% | 42.02% | 28.62% | 46.18% | 62.49% |
| SE | 0.00% | 83.26% | 71.32% | 72.28% | 98.16% |

The result is mixed. HE recall improves while precision falls; its false-positive area rises from 37,476 to 200,102 original-image pixels. EX recall also improves but its Dice decreases and false positives rise from 146,635 to 404,740 pixels. MA remains all-background. The SE result is fitting **one positively annotated training image**, so it cannot establish generalization or an 80% disease-detection accuracy. The original-grid per-image confusion counts and the unchanged 0.5 threshold are retained. Absolute total losses from the weighted and unweighted objectives should not be compared as the same quantity.

The weighted comparison supplies evidence that objective imbalance contributes to some failures, but does not isolate the cause of persistent MA failure. The separate longer-budget and representation ablations remain necessary. Do not select final research settings from these four images alone.

## Execution status at this snapshot

Two of nine core comparisons are complete. The IDRiD unweighted longer-budget run is active (last observed step 208/400); the other six are queued. A separately registered FIVES patch run waits for the core programme and is not included in the nine-run count. No validation or test image is used to choose these changes. Later Drive logs may be ahead of this immutable snapshot.

- [Machine-readable summary](snapshot/summary.json) and [metrics CSV](snapshot/metrics.csv).
- [Frozen core contract](contract-locked.json), [patch contract](patch-contract.json), and [execution status](programme-status.json).
- [Weighted original-grid per-image counts](idrid-weighted/original-grid/metrics.json), [training history](idrid-weighted/history.jsonl), and [completion receipt](idrid-weighted/status.json).
- [Baseline original-grid per-image counts](idrid-baseline/original-grid/metrics.json), [training history](idrid-baseline/history.jsonl), and [completion receipt](idrid-baseline/status.json).
- [Evidence checksums](evidence-checksums.json). Raw source images, mask exports and checkpoints are kept in the project Drive, not this publication bundle.

## Numerical validation and limits

Five CPU CTest cases and seven T4 CTest cases passed, including finite-difference derivatives, CPU/CUDA forward/backward/update parity and same-backend checkpoint resume. CUDA checks were executed on the frozen source above. Later CPU checks additionally verify that the two ablations preserve shared initial weights and initial outputs. Synthetic training and patch reconstruction tests are engineering checks, not medical evidence.

The native exporter supports OTFID, FIVES and IDRiD. This snapshot contains only the two IDRiD objective comparisons; it provides no new toxoplasmosis, glaucoma or hypertension-diagnosis result. Vessel segmentation maps visible vascular structure, not a hypertension diagnosis. Pixel areas and percentages use an explicitly stated image denominator; no physical-area or early-detection claim is established. Differentiable clDice training is still gated; the implemented hard clDice is an evaluation metric.
