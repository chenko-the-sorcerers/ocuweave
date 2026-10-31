# Changelog

Notable changes to OcuWeave are recorded here. Versions follow Semantic Versioning.
Dates use ISO 8601. Research hypotheses are not software capabilities.

## [Unreleased]

No unreleased changes recorded.

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
