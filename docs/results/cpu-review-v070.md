# CPU development review — 2026-10-31

## Execution and scope

Executed the v0.7.0 review implementation at commit `8bf815091d90d20ab426235c3c17f69576d2443e` in a Colab CPU runtime. The complete review bundle, reviewer sheets, eight overlays, preliminary technical visual notes, nine-run template and LaTeX snapshot are saved in the private Drive project. No training was performed. The historical manifest was used; the new development-group freeze and all nine exploratory runs remain pending.

- Manifest SHA-256: `b8cc4f42779d57ad5ad0fa18704730edaf62650d5594996ffab1cc6d25cb980f`.
- Archive SHA-256: `69c0c4341933cd466d042f862998dbe4830b6b73ab9270bb84f296e90a0f9404`.
- Both source hashes were checked before and after execution.
- Readable image–mask pairs: 238/238 development samples. Pair failures: 0.
- Quality flags: 104/238; all retained. Automatic exclusions, annotation edits and split changes: 0.
- Positive mask unions: 126; small positive unions (at most 1% of original full-image area): 13.
- Test image payloads were not decoded; whole-archive integrity checks read archive bytes only.

## Reference preprocessing, not model performance

| Input size | Positive mean roundtrip Dice | Small-union mean roundtrip Dice | Whole masks vanished | Original components lost by sampling |
|---|---:|---:|---:|---:|
| 192 | 0.9767266144 | 0.9250862722 | 0 | 0 |
| 384 | 0.9879767566 | 0.9618751623 | 0 | 0 |

Masks are nearest-neighbor resized, cropped to valid content and restored to original dimensions. These measurements describe reference preservation. They do not predict the benefit of 384 px training. Four-connected components are annotation topology, not clinically verified lesion instances. Private per-image CSVs also record signed area error, exact zero-tolerance boundary Dice, content size, all four padding widths, padding fraction and mask-in-padding counts.

Mean exact-boundary Dice over positive masks was 0.1386230618 at 192 and 0.2543922399 at 384; the zero-tolerance definition is deliberately strict and is not a clinical boundary metric. Mean absolute relative area error was 0.391663% and 0.179007%, respectively. Padding occupied 3.125%–44.270831% of the square canvas at both sizes. Total foreground pixels in padding: zero at either size.

## Preliminary technical visual review

Eight panels were inspected: three ranked by flag count, three by smallest positive union and two by lowest flag count (with documented deduplication). This is a purposive subset, not visual or clinical review of all 238 images.

Observed review priorities included dark/uneven illumination, diffuse blur or haze, reflection-like bright regions, and reference regions touching the captured field boundary. Two small-annotation panels warrant explicit semantic/provenance review: a thin peripheral annotation and tiny islands near the optic disc. Technical geometry agreement cannot establish that these are the correct or complete clinical targets. No mask correction was made. Empty-reference controls generated no reference contours, which does not establish clinical normality.

The eight notes identify the reviewer as AI-assisted technical visual review. Clinical lesion correctness remains unreviewed. Every image retains the disposition `retain_pending_review`; ophthalmologist adjudication and patient/eye/visit verification are still required.

## Historical pilot visualization

Three validation examples were selected before inference by original reference area (smallest positive, upper median positive) and stable identifier order (first empty-reference image). Inference used the historical query checkpoint, seed 42, input 192, selected epoch 10 of a 20-epoch budget. It is not the local-detail LQNet model.

Checkpoint SHA-256: `9fb57cf0206616ca34f3436550cced1766c4c677c990b3a306e7b2954292fe6e`.

| Selection | Original-grid Dice | IoU | False-positive full-image area |
|---|---:|---:|---:|
| Smallest positive union | 0.000000 | 0.000000 | 1.448059% |
| Upper median positive union | 0.515171 | 0.346957 | 1.979348% |
| Empty-reference control | 0.000000 | 0.000000 | 1.554268% |

Probabilities are cropped and bilinearly restored before thresholding at 0.5. The examples document missed reference extent, partial overlap and false-positive regions. These are per-image development illustrations, not cohort metrics or independent evaluation. The baseline checkpoint used validation for selection; patient independence and clinical annotation correctness remain unverified. No new training occurred.

## Verification and remaining work

52 Python tests passed, including known-value Dice/IoU/empty-mask/padding/original-grid checks. The full local run initially skipped two native tests; those passed after building the native library in a separate parity run. CMake/CTest, synthetic manifest audit, wheel packaging and CI passed at the implementation commit. The four-page LaTeX draft compiled successfully and all pages were visually inspected; no overfull equation remains. The LaTeX methods match code and distinguish current implementation from future EMR work.

The nine-row template contains configurations and pending status; unmeasured metrics, runtime and memory remain null. Freeze the provisional development groups, lock the exact manifest/configuration hashes, then run the GPU smoke gate and the nine prespecified experiments when compute becomes available. Old test results remain preserved and are not used to choose changes.
