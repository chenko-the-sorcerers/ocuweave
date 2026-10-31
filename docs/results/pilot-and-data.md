# Acquired data and first segmentation pilot

Measured 2026-10-31. Software v0.2.0, commit
`d3e8756595a728e769cc0241c9f540920140a535`. See [register](README.md) for artifact locations.

## OTFID v3

Official source [1], CC BY 4.0. Archive: 272,613,880 bytes;
MD5 `5ec60ef33e6b4841e1766c0a45ff3c7c`;
SHA-256 `69c0c4341933cd466d042f862998dbe4830b6b73ab9270bb84f296e90a0f9404`.

| Observed quantity | Count |
|---|---:|
| Images inspected | 412 |
| Mask files inspected | 179 |
| Healthy / inactive / active image labels | 132 / 188 / 92 |
| Eligible binary-segmentation images | 280 |
| Eligible healthy / inactive / active | 132 / 118 / 30 |
| Nonhealthy images excluded for missing masks | 132 |
| JPEG nonbinary-mask warnings | 173 |
| Exact decoded-pixel duplicate groups | 0 |
| Coarse dHash candidate pairs | 626 |
| Candidate pairs meeting additional RGB similarity criterion | 0 |
| Frozen train / validation / test | 196 / 42 / 42 |
| Positive train / validation / test | 104 / 22 / 22 |
| Healthy train / validation / test | 92 / 20 / 20 |

Active/inactive masks are unioned; this endpoint does not classify activity.
Masks use the frozen threshold rule (>127 for 8-bit nonbinary masks; >0 for binary
0/1 masks). Warnings do not establish clinically inaccurate masks. No missing
positive mask is treated as healthy. An earlier coarse-hash audit was superseded
before training; only the binary-v2 manifest was used. Filenames are not verified
patient identifiers; unknown patient overlap remains a material limitation.

Frozen manifest SHA-256:
`b8cc4f42779d57ad5ad0fa18704730edaf62650d5594996ffab1cc6d25cb980f`.

## Frozen training recipe

Seed 42; 20 epochs; 192×192 aspect-preserving letterbox; batch 8;
width 24, embedding 48, eight queries; AdamW lr 0.001, weight decay 0.0001;
cosine learning-rate schedule; gradient clipping 1; masked BCE + soft Dice;
training-only horizontal flips. Both models train from random initialization.
The FCN shares the encoder/pixel-decoder design, but parameter counts are not
matched. Neither network uses U-Net. Checkpoints are selected by validation
positive-image Dice; threshold 0.5 is fixed.

## Observed pilot performance

Scores below are measured on the **192-pixel letterboxed grid**, ignoring padding.
Positive Dice averages only images with lesion masks. Any-pixel FP is the fraction
of healthy images containing at least one predicted lesion pixel; it is not a
clinical false-diagnosis rate.

| Measure | Query model | FCN ablation |
|---|---:|---:|
| Best checkpoint epoch | 10 | 6 |
| Validation positive-image Dice (n=22) | 0.4805112264 | 0.4525843915 |
| Test positive-image Dice (n=22) | 0.3852694794 | 0.3734200387 |
| Test global Dice (42 images) | 0.4542774679 | 0.4104082143 |
| Test healthy images with any FP (n=20) | 13/20 (65%) | 18/20 (90%) |

This single-seed feasibility pilot does not establish superiority or clinical
utility. The test results are already observed and must not drive new design
choices. Future development uses training/validation and requires independent
held-out evidence for stronger generalization claims. No paired EMR, temporal
progression, visual-loss forecast, or disease-origin localization was trained.

## Preserved artifact hashes

| Artifact | SHA-256 |
|---|---|
| Query best.pt | `9fb57cf0206616ca34f3436550cced1766c4c677c990b3a306e7b2954292fe6e` |
| FCN best.pt | `a25392152e1f5756616584c5803c569ed8037d59385feba8710781561b1a22f5` |
| query-test.json | `42a1a5aac443816cb4ed08c87411249f03ef8e283851674c389ca650b90d8f42` |
| fcn-test.json | `0a94ce5aa157d0600ea8ea41a24797e4f3d3b38a8c3887877145f8dc06191f2c` |

The quality review rechecked that both saved test-result files were unchanged.
Complete precision, bootstrap intervals, per-image results, learning curves and
recorded timing remain in the original JSON/history files; this public table is
an explicitly selected summary, not a replacement for the complete private record.
No historical peak-memory measurement is claimed.

## References

[1] OTFID contributors, “Ocular Toxoplasmosis Fundus Images Dataset,” ver. 3,
Zenodo, record 13883485. [Online]. Available: https://zenodo.org/records/13883485.
