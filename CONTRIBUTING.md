# Contributing to OcuWeave

Use a focused issue or pull request with a reproducible command, software version, expected behavior and observed behavior. Remove private paths and identifying clinical information from logs. Do not commit medical images, patient notes, model weights or generated run artifacts.

Install `.[research]`, run `python -m unittest discover -s tests -v`, and exercise the CLI demo when changing the public workflow. Changes to geometry, objectives or metrics require tests with known expected results. Run CMake/CTest and native parity when changing C++. Keep notebooks as launchers for package code.

Document algorithmic changes separately from claims of improvement. A new architecture needs controlled validation, not just a successful forward pass. Preserve frozen manifests and prior results. Describe dataset licensing and patient/group split assumptions for each adapter. Never infer clinical correctness from a technical audit.

Follow [release procedure](docs/protocols/releases.md); update version metadata, dated changelog and relevant documentation together. Use English in public documentation. No contribution implies affiliation with or endorsement by another research institution.
