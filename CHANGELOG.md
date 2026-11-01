# Changelog

Notable changes to OcuWeave are recorded here. Versions follow Semantic Versioning.
Dates use ISO 8601. Research hypotheses are not software capabilities.

## [0.13.0] — 2026-11-01

### Added
- Standalone native disease classifier using the LQNet residual encoder, global-average pooling, stable multiclass cross entropy with analytic backward, deterministic minibatches and AdamW checkpoints.
- FIVES classification preparation with official disease labels, frozen train/validation groups, source checksums, test exclusion and independent prediction recounts (accuracy, balanced accuracy, macro F1, per-class sensitivity/specificity and AUROC).
- Registered validation targets and a Kaggle launcher with hardware checks, release/source provenance, native numerical tests and explicit real-data prerequisites. Preserve frozen segmentation experiments and their historical results.
- Repository author/branch agreement matching the user's requested Git identity and `main` workflow.

### Validation and execution scope
- Six native CPU CTest cases, 92 Python tests and six Python/native parity checks passed. Release metadata, structural manifest auditing, wheel packaging and all three installed-wheel demos passed; Kaggle notebook schema and code-cell syntax validated.
- New classifier CUDA parity is registered but not executed locally. Medical classification training is not started; Kaggle GPU/internet are blocked by account phone verification and Colab reports its GPU usage limit. No accuracy target is claimed as achieved.

## [0.12.0] — 2026-11-01

### Added
- Framework-free C++17 LQNet forward/backward, observed-label BCE/soft Dice with optional positive weights, AdamW, versioned checkpoint/resume and standalone training/inference executables.
- Optional custom CUDA convolution/matrix-product acceleration; other operators run on the host. No cuDNN, cuBLAS or ML framework is linked by the native executable.
- Separate full-resolution local-detail and spatial-query ablations; training-only tensor exports with checksum/annotation checks, original-grid evaluation, area summaries and hard clDice evaluation.
- Register paired native baseline, weighted-loss and extended-budget comparisons. Preserve historical PyTorch results and distinguish the new initialization/backend from those runs.

### Validation and scope
- Five native CPU and seven pinned-source T4 CTest cases pass, including numerical derivatives, CPU/CUDA update parity and exact checkpoint resume. 89 Python tests cover source-data contracts, native patch training/inference, reconstruction geometry and report recounts; installed-wheel legacy demos remain supported.
- Record the first completed native IDRiD training-only baseline: original-grid Dice 0.00% MA, 46.87% HE, 42.58% EX and 0.00% SE. The paired weighted run measures 0.00%, 54.01%, 42.02% and 83.26%, respectively, with increased HE/EX false positives; SE has one positively annotated training image. Subsequent comparisons remain running/queued at the recorded snapshot.
- Queue a separately registered FIVES patch diagnostic with positive/background crops and matched tiled inference. Add immutable numerical reporting and temporary-file checkpoint replacement.
- No clinical accuracy or early-detection improvement is implied. Differentiable topology loss, clinical ROI/physical area and longitudinal prediction remain outside the implemented scope.

## [0.11.1] — 2026-11-01

### Measured research and documentation
- Record two fixed-budget, training-only T4 diagnostics using source v0.11.0: four IDRiD and four FIVES images, 384 px, seed 42, 200 steps. CUDA execution, parameter updates, checkpoint reload and independent saved-mask metric recounts pass; both tasks fail the prespecified small-set fitting goal.
- Preserve per-class probability/loss/gradient traces and original-grid metrics, including zero MA/SE overlap. Record label-resize roundtrip measurements at 192 and 384 px, explicitly not a model-performance ceiling.
- Register an objective-only follow-up using observed training-label statistics; mark it unexecuted. Keep historical validation results and the main experiment matrices unchanged.
- Update the current readiness status and correct stale text/fusion module descriptions. No model architecture, loss implementation or historical checkpoint is changed.

### Validation
- Release metadata checks; 84 Python tests; 2 CTest cases; 6 native/reference parity checks; structural synthetic manifest audit; and all three installed-wheel CPU demos passed. GPU measurements are traceable to the unchanged v0.11.0 source commit.

## [0.11.0] — 2026-11-01

### Added
- Installed task commands for OTFID, FIVES and IDRiD: preparation, exact configuration generation, fixed-step training, interrupted-session continuation, validation and per-class inference.
- Periodic atomic model/optimizer/scheduler/RNG checkpoints, immutable session outputs, manifest binding, completion receipts and strict task/configuration/version/device resume guards.
- Executable training-only diagnostics: fixed-subset fitting, observed-label BCE/Dice, foreground/background probabilities, per-module gradients/updates, original-grid masks, checkpoint reload and resource records.
- Paired step-seeded augmentation and full-coverage overlapping tiled inference. Patch-based training remains pending.
- Experimental byte-Transformer note encoder, train-only structured-feature normalization, availability-aware evidence fusion and an OcuWeave tensor composition. Synthetic tests only; no pretrained LLM or clinical EMR training.
- C++ binary confusion counts with Python Dice/IoU bindings and analytical/randomized parity checks.
- Environment doctor and self-contained task/multimodal demos; installed-wheel CI exercises public commands without a source checkout.

### Updated
- Replace stale placeholder module documentation, add executable task recipes and a workflow guide, and refresh architecture/API/model-card scope.
- The compatibility `scripts/evaluate.py` launcher now defaults to original-grid validation through the public CLI. Historical input-grid test evaluation requires explicit flags.
- Preserve all historical results, source pins and frozen experiment protocols. No medical-data training, GPU readiness completion or accuracy improvement is claimed by this release.

### Validation
- Python regression and synthetic integration suite; exact CPU interrupted-run equivalence, missing-label gradients, train-only normalization, modality masking, original-grid geometry and CLI paths.
- CMake/CTest, Python/native parity, synthetic manifest audit, release metadata checks and installed-wheel CPU demos outside the checkout.

## [0.10.1] — 2026-11-01

### Documentation
- Record a planned GPU readiness gate before further main research training: tensor/label review, CUDA smoke checks, fixed training-subset fitting, probability/loss/gradient diagnostics and separately registered sampling/objective comparisons.
- Link the checklist from the README and cross-task protocol; preserve existing measured results and frozen run contracts. No GPU allocation or new medical-data training is performed in this update.

### Validation
- Version and documentation-link consistency, Python/native regression checks, synthetic manifest audit and installed-wheel CLI smoke validation.

## [0.10.0] — 2026-11-01

### Added
- FIVES official-release pairing, immutable development manifests, file integrity checks and conservative similarity grouping while preserving the official test partition.
- A separate three-variant feasibility runner with fixed step budgets, random initialization, original-grid vessel metrics, checkpoint reload checks and deterministic example selection.
- Hand-computed tests for centerline/overlap metrics, split preservation, tampering and image–mask geometry.
- Four-channel IDRiD adapter, shared-encoder query/LQNet models, annotation-availability-aware loss, per-class original-grid evaluation and a dedicated checkpoint inference API/command.
- Data-derived publication bundles with completion receipts, full per-image metrics and licensed display overlays; architecture/API documentation now reflects implemented modules.
- Explicit FIVES provisional-development opt-in after the initial audit detected cross-official-split similarity candidates; the official test remains ineligible for performance claims.

### Fixed
- Decode IDRiD indexed TIFF labels directly; do not interpret palette display colors as label values. Support the single opaque red/black RGBA mask with strict color/alpha checks. Regression fixtures cover both acquired formats and reject ambiguous values.

### Updated
- Describe the method directly as OcuWeave–LQNet in the README.
- Keep the supplementary CPU feasibility protocol separate from the uncompleted core GPU programme; REFUGE2 requires dataset access; IDRiD uses a provenance-documented CC BY redistribution.

### Measured research
- Six supplementary CPU runs completed: three FIVES vessel runs and three IDRiD four-channel lesion runs, each seed 42 and 100 steps. Publish original-grid validation receipts, histories, compute records and attributed overlays; preserve core GPU status at 0/45.
- FIVES LQNet macro Dice is 0.2022; the thin-centerline measure does not improve over Query 384. IDRiD Dice/IoU remain zero because all annotated validation channels are predicted as background at threshold 0.5. These failures remain visible.

### Validation
- 73 Python tests, native CTest/parity, installed-wheel inference checks and the complete CLI synthetic demo outside the checkout. GitHub Actions passed for the implemented code.
- Publication scores checked against original per-image confusion counts; displayed assets checked against recorded SHA-256 values. No test inference or multimodal training performed.

## [0.9.0] — 2026-11-01

### Added
- Deterministic cross-task experiment design generator: 36 core and nine conditional rows, paired controls, explicit acquisition/implementation gates, seven evaluation conditions and null unmeasured metrics. This does not execute training.
- FIVES, IDRiD and REFUGE2 dataset cards and a staged protocol for per-class targets, missing-mask handling, task-specific evaluation, robustness, compute budgets and publication of verified results.
- Register tests for paired comparisons, absent-result semantics, conditional controls and preservation of existing outputs.

### Validation
- 61 Python tests, native CTest/parity, synthetic manifest audit and installed-wheel demo. Experiment register JSON/CSV generated and checked; no new patient-data training performed.
- The frozen v0.8.1 OTFID suite remains unchanged at 0/9 completed; new-task adapters, multi-output trainers and objective terms are not claimed implemented.

## [0.8.2] — 2026-11-01

### Research record
- Document the executed v0.8.1 development freeze: 197 training, 41 validation, all 42 historical test rows preserved and all 104 quality-flagged development images retained.
- Record the nine locked configurations at 500 optimizer steps per run and the three passed CPU smoke checks, including losses, gradient norms, checkpoint reload checks and timings. Research training remains 0/9 because Colab rejected GPU allocation.
- Add instructions for continuing the existing frozen suite with its original source/data contract after a runtime reset.

### Fixed
- Update the clean exploratory launcher from the stale v0.6.0 pin and remove historical GPU/mount failures presented as current template facts.
- Include newer exploratory and CPU-review sessions in consolidated measured-report selection; retain hashes, row redaction and missing metrics.
- Synchronize package, native, citation and README version metadata. Training/model mathematics are unchanged; the already locked suite remains pinned to v0.8.1.

### Validation
- 58 Python checks, native CTest/parity, synthetic manifest audit, notebook syntax/schema, and installed-wheel demo outside the checkout.
- The actual Colab v0.8.1 preparation separately passed 57 Python checks and native CTest; these are software/pipeline checks, not new segmentation performance results.

## [0.8.1] — 2026-10-31

### Fixed
- Replace the README Mermaid embed, which failed to render in the GitHub review, with an accessible self-contained SVG architecture diagram.
- Keep public installation instructions, citation and package/native version metadata aligned with the patch release.

### Validation
- GitHub README visual inspection, Python regression suite, native CTest, packaging and installed-wheel smoke checks.

## [0.8.0] — 2026-10-31

### Added
- Installable `ocuweave` and `python -m ocuweave` commands for config creation, synthetic demo, OTFID preparation, training, validation and prediction.
- Deterministic geometric-shape fixtures and a complete CPU audit/train/checkpoint/original-grid evaluation/prediction smoke workflow, independent of notebooks and patient data.
- Public getting-started/API/contribution guides and a reorganized README with runnable commands, architecture map, experiment status and software citation.
- CLI regression checks and installed-wheel smoke validation outside the source checkout in CI.

### Fixed
- Share original-grid probability restoration between evaluation and inference; add reusable prediction artifact export.
- Record package version and dirty-tree state; resolve source commits from the package checkout instead of an unrelated caller's Git directory.
- Synchronize Python, CMake, citation and README software versions.

### Research status
- No new medical-data training or clinical result is claimed. Synthetic smoke runs are software checks; the controlled nine-run matrix and multimodal/disease-specific extensions remain pending.

## [0.7.0] - 2026-10-31

- Add a CPU-only development review bundle with photo-mask reviewer sheets, quality
  flags, reference overlays, component sampling loss, boundary/area and padding diagnostics.
  Flagged images are retained; clinical adjudication remains pending.
- Extract and test original-grid probability restoration without changing interpolation
  or threshold policy. Add hand-computed metric oracles and review integrity tests.
- Give all nine experiment rows a stable report schema including data/code/checkpoint
  hashes, configuration, metrics, timing, memory and status; unmeasured values stay null.
- Add a LaTeX introduction and methodology with equations matching the implementation,
  six verified IEEE-style references, and explicit image-only/current versus future-EMR scope.
- Document the executed 238-pair CPU audit, eight preliminary technical visual reviews,
  and three deterministically selected historical-query validation visualizations.
  No new training, exclusions, or clinical adjudication were performed.
- Synchronize CMake project metadata with the Python software version.
- Validation: 52 Python tests passed (the two native tests initially skipped in the
  full run passed in a separate native-parity run); CMake/CTest and synthetic audit passed.
  No GPU training or clinical validation is claimed by this release.

## [Unreleased]

No unreleased changes recorded.

## [0.6.0] — 2026-10-31

### Added
- Immutable development manifest revision with provisional near-duplicate family grouping; historical test rows and quality-flagged images are retained.
- Locked nine-run, paired-seed protocol for Query 192, Query 384 and OcuWeave–LQNet 384 from random initialization.
- Three-batch smoke checks for finite loss/gradients, paired geometry, checkpoint reload, timing and GPU memory; CPU checks cannot satisfy the CUDA execution gate.
- Explicit positive-image IoU, original-grid validation and source/configuration/hash guards.

### Validation
- 43 Python tests, CMake/CTest, native parity and synthetic manifest audit passed in CI; local grouping tests and syntax checks passed.
- Colab T4 allocation was blocked by quota; initial Drive mount and one recovery failed. Actual-data freezing, smoke checks and all nine runs remain pending; see docs/results/exploratory-training.md.

## [0.5.0] — 2026-10-31

### Added
- CPU-only cohort-specific quality flags with training-only thresholds and non-destructive review queues.
- Development annotation adjudication forms, exported patient-linkage evidence, and grayscale transformed near-duplicate screening across frozen splits for integrity only.
- Separate clause-local FairVLMed diagnosis/context flags and semantic/temporal review forms; no raw notes exported or test-note analysis.
- Reproducible cleaning notebook, source/test/checkpoint hash guards and private evidence snapshots.

### Validation
- 37 Python tests passed in Colab, including transformed duplicate detection, training-only quality thresholds, clause-local context/privacy checks and an end-to-end source-preservation fixture. CMake/CTest, native parity, wheel packaging, synthetic manifest audit, notebook schema/syntax and local compilation passed.
- Executed all six CPU notebook cells: 104/238 OTFID and 2,574/8,000 FairVLMed development images have technical quality flags; 6,886/8,000 original notes have priority cues. Saved clinical review queues and preserved all eight protected source/evaluation artifacts.
- One OTFID train–validation near-duplicate candidate requires adjudication. FairVLMed has zero candidates among 9,582 usable descriptors, with 418 low-variance exclusions from this detector. No clinical or patient-independence clearance is implied.

### Limitations
- Technical flags do not establish gradability, clinical annotation correctness, patient independence or EMR leakage clearance. No records are automatically removed, corrected or admitted to training.

## [0.4.1] — 2026-10-31

### Documented
- Executed review of 238 development image/mask pairs, zero geometry/foreground-count mismatches, and original/192/384 reference-boundary panels.
- Measured mask roundtrip, boundary and area-distortion results across all positive, small-lesion and train/validation strata; explicitly separate from model accuracy.
- Completed CPU notebook execution and the subsequent T4 allocation failure due to Colab usage limits. All nine new training runs remain pending, with no invented performance or resource measurements.
- Consolidated result records, source hashes and separate semantic/temporal EMR review requirements.

### Validation
- Documentation values reconciled to saved Colab JSON outputs. v0.4.0 passed 32 Python tests, CMake/CTest, packaging and CI; no model, training or metric logic changed in this patch.
- Original test-result hashes remain unchanged; clinical boundary correctness and EMR leakage clearance remain unverified.

## [0.4.0] — 2026-10-31

### Added
- Results register covering acquisition/cleaning, frozen pilot results, development diagnostics, provenance, limitations, and private evidence locations.
- Development-only image/mask geometry review with original/192/384 boundary overlays and roundtrip distortion measures.
- Experimental local residual-detail branch on the independently implemented query model; no U-Net or pretrained parameters.
- Controlled 192/384 and local-detail ablations with three paired seeds, unchanged optimizer budgets, original-grid validation, per-run resource telemetry and hash-verified completion receipts.
- Colab development notebook, explicit pending/incomplete status, documentation snapshots and private artifact inventories.
- Separate original-note glaucoma protocol and semantic/temporal leakage review form; no cross-dataset patient pairing.

### Validation
- 32 Python tests passed in Colab, including native parity, zero-residual initialization/gradient flow, original-grid evaluation, test-image exclusion and completed-artifact integrity.
- CMake/CTest, synthetic manifest auditing, wheel packaging, notebook schema/code syntax, and local compilation checks passed.

### Limitations
- New architecture implementation is not evidence of improvement, novelty, or clinical validation. Unexecuted experiments remain pending.
- No clinical boundary adjudication or cleared multimodal EMR training is claimed; test results remain frozen.

## [0.3.0] — 2026-10-31

### Added
- Pickle-free numeric image audit for FairVLMed, bounded array allocation, exact-pixel duplicate checks, CSV/NPZ label reconciliation, and immutable reports.
- Development-note review flags with explicit semantic limitations; test note contents and generated summaries are excluded from this review.
- Training/validation-only checkpoint diagnostics, lesion-area/source strata, resizing-loss checks, and deliberately selected failure panels. Test splits are rejected before loading data.
- CPU-compatible Colab quality-review notebook that preserves existing pilot checkpoints and test results.

### Validation
- 28 Python tests passed in Colab, including native parity; CMake/CTest, wheel packaging, notebook schema and code syntax validation passed.
- Synthetic corruption, duplicate, label-mismatch, object-array rejection, note privacy, and test-split protection tests.
- End-to-end synthetic checkpoint diagnostics render figures alongside the existing training/inference integration test.

### Limitations
- Numeric integrity and lexical review flags do not establish clinical gradability, patient independence, or absence of target leakage.
- No new model architecture, training result, clinical efficacy, or multimodal capability is claimed by this release.

## [0.2.0] — 2026-10-31

### Added
- Official OTFID v3 and separately licensed FairVLMed acquisition with release, license, size and checksum verification.
- Aggregate-only FairVLMed filename-pairing and clinical-table audit that preserves original notes and does not unpickle NPZ objects.
- Immutable audit manifests, missing-mask exclusions, decoded-pixel deduplication, conservative perceptual grouping, and explicit exploratory splits when patient IDs are unavailable.
- Independently implemented compact encoder and query mask head without U-Net; matched encoder/pixel-decoder FCN ablation.
- Padding-aware BCE + Dice training, validation-only checkpoint selection, held-out metrics, provenance, and original-geometry inference.
- Version-pinned Google Colab notebook with Drive storage, QA panels and learning curves.
- Synthetic integration, gradient, missing-evidence, metric and grouping tests; CPU research tests in CI.

### Validation
- 23 Python tests, CMake/CTest, and Python/C++ parity passed in Colab; CPU CI passed on the experiment branch.
- OTFID original archive checksum verified; 412 images inspected and 280 retained for the binary-mask task.
- FairVLMed archive and CSV checksums verified; all 10,000 filename pairs matched, with aggregate-only note checks.
- Coarse retinal hash false positives were visually reviewed and constrained by an RGB similarity check before freezing splits.

### Limitations
- Image-only binary lesion pilot. No paired EMR training, activity prediction, longitudinal endpoint, or clinical validation.
- Perceptual groups do not establish patient identity; image-level results remain exploratory.
- Interrupted runs retain their best checkpoint but do not resume optimizer state.

## [0.1.1] — 2026-10-31

### Added
- Synthetic fundus and lesion-segmentation thumbnail with explicit provenance.
- Author attribution to Marchel Andrian Shevchenko, Data Sorcerers, 2026.
- Release procedure, changelog, and annotated version tags.
- Public research-page link and manuscript availability information.

### Changed
- README presentation, package author metadata, and software citation.
- Version synchronized across package metadata, module, and citation.

### Research status
- No change to the algorithm: encoders, training, model weights, and clinical results remain unimplemented.

## [0.1.0] — 2026-10-31

### Added
- Modular project layout for data, encoders, fusion, losses, training, and evaluation.
- Patient-linked sample schema and structural manifest auditing.
- Synthetic manifest and explicit model input/output contracts.
- C++17 masked weighted-mean example, C interface, and Python bridge.
- Twelve Python tests, native CTest coverage, and CI workflow.
- Dataset cards for BRSET, FairVLMed, OTFID, OLIVES, and mBRSET.
- Architecture and evaluation proposals, MIT license, and citation metadata.

[Unreleased]: https://github.com/chenko-the-sorcerers/ocuweave/compare/v0.3.0...HEAD
[0.1.1]: https://github.com/chenko-the-sorcerers/ocuweave/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/chenko-the-sorcerers/ocuweave/tree/v0.1.0

[0.2.0]: https://github.com/chenko-the-sorcerers/ocuweave/compare/v0.1.1...v0.2.0

[0.3.0]: https://github.com/chenko-the-sorcerers/ocuweave/compare/v0.2.0...v0.3.0
