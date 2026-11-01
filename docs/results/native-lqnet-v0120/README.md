# Native LQNet: first controlled diagnostic

**Snapshot: 1 November 2026, 06:03 UTC.** This is a training-only engineering study on four frozen IDRiD images. It is not validation, independent testing, an early-detection study, or a comparison with the historical PyTorch initialization.

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

## Execution status at this snapshot

One of nine core comparisons is complete. The IDRiD weighted-objective run is active (last observed step 133/200); the other seven are queued. A separately registered FIVES patch run waits for the core programme and is not included in the nine-run count. No validation or test image is used to choose these changes. Later Drive logs may be ahead of this immutable snapshot.

- [Machine-readable summary](snapshot/summary.json) and [metrics CSV](snapshot/metrics.csv).
- [Frozen core contract](contract-locked.json), [patch contract](patch-contract.json), and [execution status](programme-status.json).
- [Baseline original-grid per-image counts](idrid-baseline/original-grid/metrics.json), [training history](idrid-baseline/history.jsonl), and [completion receipt](idrid-baseline/status.json).
- [Evidence checksums](evidence-checksums.json). Raw source images, mask exports and checkpoints are kept in the project Drive, not this publication bundle.

## Numerical validation and limits

Five CPU CTest cases and seven T4 CTest cases passed, including finite-difference derivatives, CPU/CUDA forward/backward/update parity and same-backend checkpoint resume. CUDA checks were executed on the frozen source above. Later CPU checks additionally verify that the two ablations preserve shared initial weights and initial outputs. Synthetic training and patch reconstruction tests are engineering checks, not medical evidence.

The native exporter supports OTFID, FIVES and IDRiD. This snapshot contains only the IDRiD baseline; it provides no new toxoplasmosis, glaucoma or hypertension-diagnosis result. Vessel segmentation maps visible vascular structure, not a hypertension diagnosis. Pixel areas and percentages use an explicitly stated image denominator; no physical-area or early-detection claim is established. Differentiable clDice training is still gated; the implemented hard clDice is an evaluation metric.
