# Changelog

Notable changes to OcuWeave are recorded here. Versions follow Semantic Versioning.
Dates use ISO 8601. Research hypotheses are not software capabilities.

## [0.10.0] — 2026-11-01

### Added
- FIVES official-release pairing, immutable development manifests, file integrity checks and conservative similarity grouping while preserving the official test partition.
- A separate three-variant feasibility runner with fixed step budgets, random initialization, original-grid vessel metrics, checkpoint reload checks and deterministic example selection.
- Hand-computed tests for centerline/overlap metrics, split preservation, tampering and image–mask geometry.
- Four-channel IDRiD adapter, shared-encoder query/LQNet models, annotation-availability-aware loss and per-class original-grid evaluation.
- Explicit FIVES provisional-development opt-in after the initial audit detected cross-official-split similarity candidates; the official test remains ineligible for performance claims.

### Fixed
- Decode IDRiD indexed TIFF labels directly; do not interpret palette display colors as label values. Add a regression fixture matching the acquired 0/1 red palette.

### Updated
- Describe the method directly as OcuWeave–LQNet in the README.
- Keep the supplementary CPU feasibility protocol separate from the uncompleted core GPU programme; REFUGE2 requires dataset access; IDRiD uses a provenance-documented CC BY redistribution.

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
