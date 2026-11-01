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
