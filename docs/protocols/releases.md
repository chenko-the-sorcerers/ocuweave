# Release procedure

1. Classify the change: patch for compatible fixes/documentation, minor for new functionality, major for incompatible changes after 1.0. During 0.x, document breaking changes explicitly.
2. Update `pyproject.toml`, `src/ocuweave/__init__.py`, and `CITATION.cff` together.
3. Add a dated entry to `CHANGELOG.md`; separate implemented capabilities from research plans.
4. Run Python tests, CMake/CTest, native parity, synthetic manifest auditing, and packaging validation.
5. Review the release contents. Exclude private data, credentials, generated outputs, and unlicensed weights.
6. Commit, create an annotated `vX.Y.Z` tag on the validated commit, and push that exact tag. Do not move published tags.
7. Verify the remote files, tag, and CI. Report pending or failed CI honestly.

Future checkpoints require their own model card, training configuration, source revision, dataset provenance, evaluation protocol, checksum, and redistribution license. A software tag does not imply a model is available or clinically validated.
