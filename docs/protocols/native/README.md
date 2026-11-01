# LQNet native: implementation and controlled diagnostics

The native executable trains the image-only LQNet graph using C++17 and explicitly implemented vector–Jacobian products, observed-label BCE/soft Dice, and AdamW. It does not import or link PyTorch, TensorFlow, JAX, LibTorch, cuDNN or cuBLAS. Every model starts from random weights unless an explicit native checkpoint is supplied.

## Dependency and execution boundary

- CPU: C++17 standard library and CMake/compiler only.
- Optional acceleration: NVIDIA CUDA toolkit/runtime/driver and our convolution/matrix-product kernels. Other operations and the differentiation tape execute on the host. This is a **mixed CPU/CUDA reference backend**, not an optimized all-GPU engine. Host/device transfers and retained activations limit throughput and memory.
- Input preparation and original-grid measurement: Python standard library plus Pillow for image decoding/resampling. No tensor/ML framework or NumPy is imported by these scripts.
- Historical Python/PyTorch experiments remain intact in the repository. Installing their optional dependencies is not required to build or run the native tools.
- CUDA numerical parity must pass on the actual device before medical-data fitting. A CPU pass alone does not verify the CUDA kernels.

## Build and smoke test

```bash
cmake -S . -B build-native -DCMAKE_BUILD_TYPE=Release
cmake --build build-native -j2
ctest --test-dir build-native --output-on-failure
./build-native/ocuweave-lqnet --synthetic true --output outputs/native-demo \
  --width 4 --dim 8 --queries 2 --steps 30 --size 32
```

For a T4, configure a separate build with `-DOCUWEAVE_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`, run its CTest suite including `native_cuda_parity`, and pass `--device cuda`. Do not use synthetic scores as medical performance.

## Frozen tensors and provenance

```bash
python scripts/native_data.py --task idrid --root /path/to/extracted/idrid \
  --manifest /path/to/frozen/manifest.json --output data/native/idrid.bin \
  --ids IDRiD_01 IDRiD_03 IDRiD_02 IDRiD_04 --size 384
./build-native/ocuweave-lqnet --batch data/native/idrid.bin --classes 4 \
  --device cpu --output outputs/native-idrid-baseline --steps 200
python scripts/native_evaluate.py \
  --predictions outputs/native-idrid-baseline/predictions.bin \
  --metadata data/native/idrid.json --root /path/to/extracted/idrid \
  --output outputs/native-idrid-baseline/original-grid
```

The exporter verifies source image/mask checksums (decoded RGB/union-mask hashes for OTFID), explicit training membership, label encoding and geometry. OTFID accepts an extracted archive root or the original ZIP, with its frozen archive hash checked. Missing masks remain unavailable, not negative. It records IDs, manifest/input hashes, image dimensions, crop coordinates and Pillow version. Batch payloads are little-endian float32 with explicit dimensions, availability and partition; checkpoint files store native weights, AdamW moments, optimizer settings and step count. These formats do not deserialize executable objects. Output directories are never overwritten.

The LQNet graph retains the earlier convolutional feature pyramid, query cross-attention, per-class mask heads and local residual branch. Native initialization uses a name-keyed PRNG; Q/K/V are explicit independent parameter tensors. Initialization and floating-point accumulation differ from historical PyTorch runs. **Native results must be compared against the paired native baseline, not attributed as improvements over the historical backend.** The previously registered v0.11.0 loss comparison remains a separate unexecuted historical contract.

## Experiment order

All proposed medical runs are training-only diagnostics on the frozen four-image selections. Start with seed 42, 384 px, width 24, embedding 48, eight queries, float32, AdamW lr=0.001, weight decay=0.0001, clipping=1, no augmentation, threshold=0.5. Report original-grid image-macro metrics; preserve failures. The engineering small-set goal is input-grid Dice >=0.80 for each positively annotated class; it is not a clinical threshold.

1. **Native baseline and weighted objective:** 200 steps each, identical source/input/initial weights. IDRiD positive weights are `[20, 11.667243314128676, 12.01800482195464, 17.261825830351288]`, derived from the previously frozen 43 training images. Soft Dice remains unchanged. Record precision/recall and FP area, not only Dice.
2. **Longer budget:** resume the unweighted baseline for 200 additional steps, for 400 total, with unchanged optimizer state/objective. Do not call a difference versus the weighted run a budget-only effect.
3. **Native-resolution patches:** exporter supports one lexical positive and one zero-annotated-positive tile per source image. Unknown classes are not certified background. This changes sampling and pixel exposure; it is not a resolution-only comparison. `scripts/native_tiles.py` implements full-image tiled reconstruction with uniform overlap averaging; a declared exposure/compute budget is still required before interpreting patch results.
4. **Detail and query separately:** `--variant detail_full` moves only the local branch to full input resolution. `--variant spatial_query` adds a zero-initialized spatial query router while preserving the baseline initial output. Compare each independently with the same unweighted native baseline; report parameters and runtime.
5. **Vessel continuity:** `--vessels true` adds hard clDice and skeleton precision/recall at the input grid using padded Zhang–Suen thinning. This is an evaluation metric, **not a topology training loss**. A differentiable clDice objective remains a separately gated experiment after baseline stability; it is not implemented by this release's optimizer.

No accuracy increase is assumed. No validation/test is used to select these changes. Broader development experiments remain pending until diagnostics are reviewed. Patient identity and clinical annotation correctness are still unverified.

## Area and early-detection interpretation

The executable reports predicted pixel area, centroid, bounding box, and percentage of valid input image pixels. `native_evaluate.py` restores probabilities before thresholding and reports original-image areas and overlap. Neither denominator is a clinically verified retinal field; percentages are not probabilities of disease or estimates of visual impairment. Physical area in mm² is null without acquisition calibration. A supplied verified retinal ROI would be needed for percentage of retinal area. Longitudinal change requires registration, visit linkage and acquisition normalization. Early detection needs stage/onset labels, a defined clinical endpoint, appropriate reference standards and independent patient-level evaluation; segmentation overlap alone cannot establish it.

## Validation requirements

Analytical BCE/Dice and AdamW oracles; finite-difference derivatives for convolution, matrix products, normalization, attention composition, interpolation and weighted loss; missing/padded supervision checks; exact same-backend checkpoint resume; CPU/CUDA forward/backward/update parity; input/output geometry and metric oracles. CPU and CUDA summation orders differ, so cross-backend parity uses declared tolerances, not bitwise equality. New baselines must use one backend throughout.

## Tiled inference and area reconstruction

```bash
python scripts/native_tiles.py --executable build-native/ocuweave-predict \
  --checkpoint outputs/native-run/latest.ckpt --image /path/to/retina.png \
  --output outputs/native-tiled --classes 1 --size 384 --tile-size 384 --overlap 96 --device cuda
```

Every original pixel must be covered. Tile probabilities are restored to their crop coordinates and averaged before thresholding. The geometry oracle reconstructs a known spatial ramp exactly, including overlaps and edge tiles, and rejects uncovered pixels. Tiled inference changes the prediction context relative to full-image letterboxing; its results require their own declared protocol and original-grid evaluation. No label is used to select inference tiles.

Hard clDice is based on the skeleton-overlap definition of Shit et al., “clDice — A Novel Topology-Preserving Loss Function for Tubular Structure Segmentation,” CVPR 2021, pp. 16560–16569. [Official paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.pdf). Our metric uses our padded Zhang–Suen thinning implementation; it is not the paper's differentiable soft-skeleton training objective.

## Execution and reporting

`configs/native/diagnostic-programme.json` declares nine small-set runs: five IDRiD and four FIVES comparisons. The running native source is frozen independently of the final software release; `scripts/native_programme.py` checks that revision and batch hashes before execution. Do not restart the launcher into existing output directories. A manual interrupted-run continuation must use a new output path and the saved native checkpoint, retaining the original experiment contract.

`configs/native/patch-programme.json` registers a separate tenth run. It fits eight 384 × 384 source-resolution crops (one annotated-positive and one zero-annotated-positive crop from each of the four FIVES training images). The selection is lexical and deliberately fixed, not optimized using validation. Background crops can contain black image borders. Both the full-image-trained checkpoint and the patch-trained checkpoint then receive identical full-coverage tiled inference with 96-pixel overlap. This separates inference-context effects, but does not isolate resolution from changed training context, sampling, or exposure. `scripts/native_patch_programme.py` queues it after the core programme and preserves failure receipts.

```bash
python scripts/native_report.py --session /path/to/native/session --output /path/to/new/report
```

Pass `--watch` with a new report-root directory to save a fresh snapshot after each run-state transition until the core programme completes or reports failure. This background writer does not restart training.

Reports include only completed-run metrics; running/queued jobs retain null/unavailable results. Confusion counts independently verify Dice and IoU. Precision/recall are pooled across annotated images, whereas the primary Dice/IoU are positive-image macro averages; these aggregations must not be interchanged. Memory values are sampled process usage, not allocator peaks. [Measured native evidence and execution status](../../results/native-lqnet-v0120/README.md).

Checkpoint writes in the released source use a same-directory temporary file and rename. The already running development source predates this I/O safeguard; its numerical graph is unchanged. Checkpoint bytes include AdamW state. A successful reload does not establish clinical accuracy.
