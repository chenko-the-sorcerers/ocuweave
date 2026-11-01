# OcuWeave working agreement

- Work on `main` for this repository. Do not create branches with the `codex/` prefix; the user explicitly rejected those names on 2026-11-01.
- New commits use `Marchel Andrian Shevchenko <icshevhenko17@gmail.com>`. The user confirmed that the authenticated GitHub account `chenko-the-sorcerers` is the requested account. Do not use a different connector account for mutations.
- Preserve historical commit hashes and release tags; the user clarified that branch names, not research provenance, were the issue.
- Follow `docs/protocols/releases.md`: semantic versioning, dated changelog, synchronized metadata, relevant tests, tagged release and CI verification.
- Implement new native learning code in C++/CUDA without an ML framework or pretrained weights. Historical Python/PyTorch results remain a separate preserved backend.
- Research targets: segmentation validation Dice >0.80 and disease-classification validation accuracy >0.85. These are objectives, not promises or measured results. Report per-class metrics and failures; never use test labels to tune changes.
- Preserve frozen running experiments. Register new classification/segmentation experiments independently, retaining data hashes, source revisions, seeds, checkpoints and honest execution status.
