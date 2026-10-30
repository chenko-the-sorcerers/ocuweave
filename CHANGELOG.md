# Changelog

Notable changes to OcuWeave are recorded here. Versions follow Semantic Versioning.
Dates use ISO 8601. Research hypotheses are not software capabilities.

## [Unreleased]

No unreleased changes recorded.

## [0.2.0] — 2026-10-31

### Added
- Official OTFID v3 acquisition with release, license, size and checksum verification.
- Immutable audit manifests, missing-mask exclusions, decoded-pixel deduplication, conservative perceptual grouping, and explicit exploratory splits when patient IDs are unavailable.
- Independently implemented compact encoder and query mask head without U-Net; matched encoder/pixel-decoder FCN ablation.
- Padding-aware BCE + Dice training, validation-only checkpoint selection, held-out metrics, provenance, and original-geometry inference.
- Version-pinned Google Colab notebook with Drive storage, QA panels and learning curves.
- Synthetic integration, gradient, missing-evidence, metric and grouping tests; CPU research tests in CI.

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

[Unreleased]: https://github.com/chenko-the-sorcerers/ocuweave/compare/v0.2.0...HEAD
[0.1.1]: https://github.com/chenko-the-sorcerers/ocuweave/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/chenko-the-sorcerers/ocuweave/tree/v0.1.0

[0.2.0]: https://github.com/chenko-the-sorcerers/ocuweave/compare/v0.1.1...v0.2.0
