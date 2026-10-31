# OTFID exploratory development preparation — 31 October 2026

Author: Marchel Andrian Shevchenko. This record describes preparation and pipeline checks, not model performance or clinical validation.

## Provenance and cohort decision

The execution used OcuWeave v0.8.1, commit `afd6b45c16761fba8cfc214c921cae87652e47c9`, in Colab with PyTorch `2.11.0+cpu`. The controlling private session is `02 — Audits/exploratory-v0.8.1/20261031T165504Z` within the existing project Drive. An initial setup attempt inherited the earlier notebook's source pin; its logs were retained separately. The final session reran setup and all software checks against the exact v0.8.1 commit before freezing data.

| Evidence | SHA-256 |
| --- | --- |
| Historical manifest | `b8cc4f42779d57ad5ad0fa18704730edaf62650d5594996ffab1cc6d25cb980f` |
| Frozen development manifest | `56c4721a3063cfb5e5fa89e909fbb47494fccc6c4b9f1597d973e1b3c3fe941e` |
| Locked experiment protocol | `457c124f706ec803eab8940ecaced3781b0dc8ca45fe0d2c0493723070a10fa3` |
| Source OTFID archive | `69c0c4341933cd466d042f862998dbe4830b6b73ab9270bb84f296e90a0f9404` |
| Near-duplicate candidate file | `f863a52545e828027f6a8ec70db4854ea9b50adbac4748d601d20647cb845b0d` |

| Partition | Historical | Revised |
| --- | ---: | ---: |
| Training | 196 | 197 |
| Validation | 42 | 41 |
| Historical test | 42 | 42 |

One validation image moved to training under the prespecified candidate-connected grouping rule. This grouping is provisional; it does not establish patient identity or confirm the candidate is a duplicate. All 238 development images, including 104 quality-flagged images, remain eligible. There were zero quality exclusions. Test rows were preserved exactly. Image identifiers and per-image decisions remain in private Drive.

## Locked experiment

The nine-run contract is stored in `04 — Runs/v0.8.1/20261031T165504Z/resolution-detail-seeds` in the same private project. It pins the original source commit even after documentation releases.

| Variant | Resolution | Seeds | Epochs | Batch | Optimizer steps per run |
| --- | ---: | --- | ---: | ---: | ---: |
| Query | 192 | 42, 123, 2026 | 20 | 8 | 500 |
| Query | 384 | 42, 123, 2026 | 20 | 8 | 500 |
| LQNet / query + local detail | 384 | 42, 123, 2026 | 20 | 8 | 500 |

Loss, augmentation, optimizer, scheduling, split membership and step budget are shared. All models initialize randomly. No U-Net, pretrained weights or EMR is used. The historical test is not evaluated. The [protocol](../protocols/exploratory_training.md) defines checkpoint selection, original-grid validation, small-lesion strata and limitations.

## Execution boundary

Colab refused a T4 allocation because of usage limits. CPU preparation proceeded after releasing an inactive completed review runtime. The final Colab software verification passed 57 Python tests, native CTest, and notebook syntax/schema checks.

Research training remains **0/9 completed**. There are no new research Dice/IoU scores. A future GPU session must pass a new CUDA smoke test using the same source/data contract before it can execute the existing suite. Patient identities and clinical annotations remain unverified; FairVLMed remains separate.

## CPU smoke measurements

Each variant processed three full batches of eight images at seed 42. Image/mask/padding geometry, finite loss and gradients, nonzero total gradient norms and checkpoint reload equality passed. CUDA allocated/reserved memory is **unmeasured (null)**, not zero.

| Variant | Batch losses | Gradient norms before clipping | Recorded seconds | Checkpoint reload |
| --- | --- | --- | ---: | --- |
| Query 192 | 1.135985, 1.159370, 1.243416 | 0.489448, 0.127499, 0.671652 | 7.797 | Passed |
| Query 384 | 1.136406, 1.159538, 1.242746 | 0.488686, 0.128688, 0.677491 | 6.997 | Passed |
| LQNet 384 | 1.136406, 1.159573, 1.242860 | 0.502032, 0.130947, 0.689685 | 11.676 | Passed |

These short timings include setup and checkpoint checks; they are not a controlled speed benchmark or a basis for training-duration estimates. The losses are pipeline observations, not comparative research endpoints. Smoke weights will not initialize the nine research runs.

The original archive, historical manifest, historical test result files and pilot checkpoints passed the final before/after hash comparison. A private documentation snapshot indexed 263 artifacts. The v0.8.1 snapshot inventory includes this session, but its consolidated measured-report selector still targets old versions; v0.8.2 corrects that selector for future snapshots. The session's own JSON receipts and logs remain the controlling evidence.

## Supplementary technical visual review

Two positive training examples were selected deterministically from manifest order and rendered at 192 and 384 pixels with reference contours. Neither showed an obvious gross translation or mirror mismatch. One central bright-region example was visibly blurred; the other had dim/uneven illumination. Both remained retained. All four resized pairs had zero annotated-mask overlap with padding. These observations do not confirm clinical lesion completeness. Private overlay PNGs, per-pair pixel counts and reviewer notes are saved alongside a continuation receipt; no images or sample identifiers are redistributed here.
