# Public software workflow validation — v0.8.0

Date: 2026-10-31. This record concerns software behavior, not medical-data model quality.

- 57 Python tests passed, including five new public-workflow checks: all three architecture round trips, deterministic synthetic archive creation, CLI guards, package-source provenance, and installed-command training/evaluation/prediction.
- CMake/CTest passed; the Python suite included numerical parity with the macOS C++ shared library.
- Built the wheel, installed it into a separate target directory and ran its CLI outside the source checkout using the existing test environment's dependencies. Import paths were checked to point to the installed wheel, not editable source. The CLI generated data, trained, restored original geometry, evaluated validation and saved prediction artifacts.
- The installed wheel correctly reported source commit as unavailable. Source-checkout runs record the actual package checkout and dirty state, not the caller's Git repository.
- Reference and prediction dimensions, probabilities, mask thresholding, TP/FP counts, output overwrite protection, archive determinism and untouched synthetic test split were checked.
- CI additionally installs the wheel into a separate virtual environment on Linux and executes the public CPU demo.

The local tests used PyTorch 2.11.0 on CPU. No new OTFID, FairVLMed, vessel or optic disc/cup training is part of this release. The two-epoch geometric fixture is an installation smoke test; its scores must not appear as retinal segmentation results. The nine-run medical-data matrix remains pending.
