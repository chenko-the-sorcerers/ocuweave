# Numeric integrity, note flags and development errors

Measured 2026-10-31. Software v0.3.0, commit
`6ed33aa24749b32a75d21be98258e102b05439a6`. CPU execution; no new training.
Seven notebook code cells completed without errors; 28 Python tests including
native parity, CMake/CTest and packaging validation passed. See [register](README.md).

## FairVLMed

Source: official Harvard FairVLMed [1], a glaucoma cohort with paired SLO fundus
images and clinical notes. Its CC BY-NC-ND 4.0 terms are separate from MIT software.
All 10,000 CSV filename pairs matched the archive. Official splits remain
7,000 training / 1,000 validation / 2,000 test; labels are 5,048 yes / 4,952 no.

Archive bytes: 2,668,667,218;
SHA-256 `01bbfba4d38dbda8da24e5f73ba4e20b3cbfe91ff9dd4eb03e72ef4196ff7153`.
CSV bytes: 12,296,934;
SHA-256 `d3e32b972c3eae1f3d58c3cc3caec00dc91c83b394c0f127e1032d531e226845`.

| Numeric image check | Observed result |
|---|---:|
| Decoded SLO images | 10,000/10,000 |
| Dimensions and dtype | All 664×512, uint8 |
| Numeric decode failures | 0 |
| Constant images | 0 |
| CSV/NPZ label mismatches | 0 |
| Unreferenced NPZ files | 0 |
| Exact-pixel duplicate groups, including across splits | 0 |

Only bounded numeric `slo_fundus` and `glaucoma` arrays are decoded with
`allow_pickle=False`. This does not establish clinical gradability, absence of
near duplicates, de-identification quality, or independent patient linkage.

## Original-note flags: development splits only

| Lexical flag | Training (n=7,000) | Validation (n=1,000) |
|---|---:|---:|
| Missing note | 0 | 0 |
| Diagnosis term | 4,475 | 627 |
| Suspect/risk context | 2,383 | 344 |
| Negation context | 4,763 | 688 |
| Family context | 2,411 | 331 |
| Treatment proxy | 1,919 | 283 |
| Endpoint proxy | 4,660 | 658 |

Flags overlap; their columns must not be added as counts of distinct patients.
5,102/8,000 development notes contain a diagnosis-term flag. There are 30
case/whitespace-normalized duplicate-note groups, six crossing training/validation.
An earlier structural audit found 19 *raw exact-text* duplicate groups across all
splits; the normalization and split scopes differ, so these counts are not a trend.
Repeated text may be templates and does not prove patient duplication.

These are broad co-occurrence flags, not scope-aware negation, temporal assessment
or confirmed leakage. Test-note content and GPT-generated summaries were excluded
from this semantic-review preparation. No note has been certified target-free.
Do not infer that absence of a flag makes a record safe for training.

## OTFID frozen-checkpoint development diagnostics

Input grid 192, threshold 0.5; original weights reused. Validation already selected
these checkpoints, so the numbers below are descriptive development evidence.

| Model / split | Positive n | Positive Dice | Healthy n | Healthy any-pixel FP |
|---|---:|---:|---:|---:|
| Query / train | 104 | 0.5580 | 92 | 51.1% |
| Query / validation | 22 | 0.4805 | 20 | 65.0% |
| FCN / train | 104 | 0.5122 | 92 | 76.1% |
| FCN / validation | 22 | 0.4526 | 20 | 70.0% |

| Original lesion area / full image (validation) | n | Query Dice | FCN Dice |
|---|---:|---:|---:|
| >0–1% | 5 | 0.0231009355 | 0.0380090508 |
| >1–5% | 6 | 0.5150283490 | 0.4701034070 |
| >5% | 11 | 0.6695974735 | 0.6314719015 |

No positive masks disappeared entirely at the pilot input resolution. This does
not establish preservation of their boundaries or diagnostic detail. Area is
relative to the full original image, not an annotated retinal field.

| Validation subgroup | Positive n | Query positive Dice | FCN positive Dice |
|---|---:|---:|---:|
| Active | 4 | 0.5784853433 | 0.5387456748 |
| Inactive | 18 | 0.4587392004 | 0.4334374396 |
| Hospital General Pediátrico Acosta Ñu | 9 | 0.4328103085 | 0.3814041460 |
| Hospital de Clínicas | 13 | 0.5135349387 | 0.5018630230 |

Healthy any-pixel FP by source: query 6/7 and 7/13, FCN 6/7 and 8/13,
respectively. Small strata preclude strong comparisons. Both full source/label
reports, per-image Dice/precision/recall, FP area, interval estimates and selected
failure panels are retained in private Drive. Panels intentionally show the three
lowest positive-Dice cases and the three largest healthy FP areas for each split
and model; they are not representative examples.

## Interpretation and unresolved questions

Small lesions and false-positive regions are development priorities. Resampling,
annotation boundaries, source appearance and architecture are candidate
explanations, not established causes. The next experiment isolates resolution
before testing local detail. No masks or labels have been changed. No clinical
review or new multimodal/prognostic capability is claimed.

## References

[1] Harvard AI Robotics, “FairVLMed,” official dataset card. [Online]. Available:
https://huggingface.co/datasets/harvardairobotics/FairVLMed.

[2] Y. Luo et al., “FairCLIP: Harnessing Fairness in Vision-Language Learning,”
in *Proc. IEEE/CVF CVPR*, 2024, pp. 12289–12301. [Online]. Available:
https://openaccess.thecvf.com/content/CVPR2024/html/Luo_FairCLIP_Harnessing_Fairness_in_Vision-Language_Learning_CVPR_2024_paper.html.
