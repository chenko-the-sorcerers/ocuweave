# Cleaning review: observed results

Marchel Andrian Shevchenko · Data Sorcerers · 2026. Executed 2026-10-31 on Colab CPU.

**Technical screening completed; clinical cleaning and patient-independence verification are not complete.** No image/note was deleted, mask corrected, split changed or new acquired-data training performed.

## Evidence and reproducibility

The numerical audit used commit `4d890389e4fbaecf7481ef628722bfc00142ffd8` (v0.5.0 candidate). Final v0.5.0 documentation adds the observed findings and an expanded archive inventory; the screening algorithm is unchanged.

Private evidence root: `02 — Audits/cleaning-review-v0.5.0/20261031T061228Z/` in the authorized OcuWeave project Drive. It contains `review/summary.json`, cohort-specific quality CSVs, thresholds, near-duplicate JSON, patient-linkage evidence, annotation/EMR review forms, selected previews, software-test logs and before/after hashes. The [executed notebook](https://colab.research.google.com/drive/11inGqNcX7ew7SZkQR1ANao_12lDrzkMI) retains existing private access. Raw notes and image-level identifiers are not reproduced here.

All six protocol code cells completed. Supplemental cells inspect development examples and the train–validation candidate. A supplemental main-kernel import initially failed; the retry used the pinned source directory and is recorded separately. The acquired-data audit had already succeeded in fresh subprocesses.

Validation: 37 Python tests, CMake/CTest, native parity, wheel packaging, synthetic manifest checks, notebook schema/syntax and branch CI passed. Eight protected artifacts have identical pre/post SHA-256 values: source archives/CSV, frozen manifest, two checkpoints and two historical test-result JSON files.

## Image quality priorities

Thresholds are cohort-specific training quantiles, not clinical gradability cutoffs. Flag categories overlap; column totals must not be added as distinct images.

| Technical flag | OTFID development (238) | FairVLMed development (8,000) |
|---|---:|---:|
| Low sharpness | 25 | 799 |
| Low supported intensity | 26 | 808 |
| Low contrast | 24 | 721 |
| Bright clipping | 25 | 352 |
| Large dark region | 22 | 797 |
| Non-dark field border contact | 12 | 0 |
| Intensity asymmetry / possible artifact | 23 | 799 |
| At least one flag | **104 (43.7%)** | **2,574 (32.2%)** |
| Automatically excluded | **0** | **0** |

Six selected examples per cohort (four high-flag-count and two low-flag-count controls) were visually inspected as technical previews. OTFID examples show differing illumination and vessel/detail visibility; FairVLMed examples include substantial dark or uneven-intensity regions. These observations do not identify their clinical cause or establish gradability. No clinical reviewer has signed the forms.

FairVLMed's training 90th-percentile border-contact value is 1.0, so a strict greater-than rule flags no images. This is a **saturated proxy with no ability to exceed its cutoff**, not evidence of zero cropped retina. Likewise, its bright-clipping 90th percentile is zero, so even small nonzero bright fractions trigger review. Retinal-field truncation needs a modality-aware field model or expert inspection; the current proxy cannot resolve it.

The OTFID quality scope is the frozen mask-eligible cohort. The original archive's 132 nonhealthy images without masks remain preserved and excluded under the earlier segmentation eligibility rule, not newly deleted by this audit.

## Annotation review

A clinical adjudication queue now covers **all 238 development records**, including 126 positive masks and 112 healthy empty masks. Fields cover lesion location, boundary completeness, missed regions, activity-mask union, reviewer/date, decision and rationale. All begin unreviewed. Existing original/192/384 overlays and [geometry measurements](annotation-review.md) remain linked evidence; zero geometry mismatch is not clinical correctness. No ophthalmologist review or annotation correction is claimed.

## Near duplicates and identity

| Scope | OTFID | FairVLMed |
|---|---:|---:|
| Images processed for numerical fingerprints | 280 | 10,000 |
| Low-variance descriptors excluded from this near-duplicate screen | 0 | 418 |
| Descriptors compared | 280 | 9,582 |
| Candidate pairs | **1** | **0** |
| Cross-split candidate pairs | **1: train–validation** | **0** |
| Candidate search truncated | No | No |
| Verified patient/eye/visit mapping | **Absent** | **Absent** |

The OTFID pair comprises two healthy-labelled, empty-mask images of equal dimensions. Center-normalized grayscale cosine similarity is **0.9958892465** and normalized RMSE **0.0925312564**, with the identity transform. A development-only panel shows very similar vascular layout and framing. This is a priority potential repeated-view/image-family finding, not proof of patient identity or an adjudicated duplicate. Source filenames, original images and review status remain in private evidence. No candidate in this run involves the frozen test split.

The new sensitivity to grayscale/brightness normalization differs from the earlier stricter RGB screen. Preserve both historical results and the new discrepancy. The train–validation independence assumption must remain unresolved until this pair and patient linkage are adjudicated. Any future duplicate grouping or resplitting requires a versioned protocol; preserve existing results as exploratory history and do not use test performance to choose changes.

The exported OTFID CSV contains image name, label and source only; the frozen manifest has **zero verified patient IDs**. The acquired FairVLMed CSV contains filename, demographics, original note, generated summary, glaucoma label and official split, with no explicit patient/eye/visit or timestamp column. The official FairVLMed card reports 10,000 patients/samples, but no independently verified image-to-patient mapping was supplied [1]. No patient identity was inferred from filenames, similar images or repeated notes, and no cross-cohort matching was attempted.

Zero FairVLMed candidates applies only to this detector and its 9,582 usable descriptors. Low-variance images, other crops, arbitrary rotations and longitudinally changed views remain unresolved. Earlier exact-pixel integrity checks are separate evidence; they do not close this patient-linkage gap.

## FairVLMed original-note context review

Training (7,000) and validation (1,000) original notes were processed; test-note content and generated summaries were not semantically analyzed. Counts below are notes with at least one matching cue, not diagnosis prevalence, confirmed leakage, or patient counts.

| Review cue | Training | Validation | Total |
|---|---:|---:|---:|
| Diagnosis mention | 4,475 | 627 | 5,102 |
| Nearby negation | 1,050 | 130 | 1,180 |
| Nearby family context | 780 | 98 | 878 |
| Nearby suspicion | 1,839 | 272 | 2,111 |
| Nearby history cue | 719 | 105 | 824 |
| Nearby laterality | 1,864 | 245 | 2,109 |
| Unqualified mention proxy | 2,815 | 390 | 3,205 |
| Treatment proxy | 1,919 | 283 | 2,202 |
| Endpoint proxy | 4,660 | 658 | 5,318 |
| Date token present | 2,614 | 391 | 3,005 |
| Priority review: diagnosis, treatment or endpoint | **6,039** | **847** | **6,886** |

Clause-local proximity is narrower than the earlier note-wide lexical screen, so these context counts are different endpoints. Negation, family history and dates require semantic review; a date token does not verify pre-prediction availability. All **8,000** notes have review-form rows; the 1,114 without priority cues are not cleared. No raw text is exported, no label-bearing spans are automatically removed and no note is admitted to model training.

## Decisions before subsequent experiments

1. Adjudicate image-quality candidates and controls, including the 418 low-variance FairVLMed descriptors and the saturated border-contact proxy; keep originals and record decisions.
2. Obtain ophthalmologist review of lesion location/completeness and questionable masks before clinical claims or label revision.
3. Investigate the OTFID train–validation candidate and obtain documented patient/eye/visit linkage. A draft [custodian evidence request](../protocols/patient_mapping_request.md) is prepared but has not been sent.
4. Complete FairVLMed assertion, experiencer, time-of-availability and treatment/endpoint leakage review before multimodal glaucoma training. Keep it separate from OTFID toxoplasmosis.

[1] Harvard AI Robotics, “FairVLMed,” official dataset card. [Source](https://huggingface.co/datasets/harvardairobotics/FairVLMed/blob/main/README.md). Algorithmic rules and other official references: [cleaning protocol](../protocols/cleaning_review.md).
