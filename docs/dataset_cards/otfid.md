# Ocular Toxoplasmosis Fundus Images Dataset v3

Source: https://zenodo.org/records/13883485. Accessed and audited 2026-10-31.
License: CC BY 4.0. Dataset rights remain separate from the MIT code license.

## Verified release
- Archive: `Ocular_Toxoplasmosis_Data_V3.zip`, 272,613,880 bytes.
- Official MD5: `5ec60ef33e6b4841e1766c0a45ff3c7c`.
- 412 image files and 412 CSV rows; 179 mask files.
- CSV labels: 132 healthy, 188 inactive, 92 active. These are release CSV labels,
  not a reconstruction of the paper's activity categories.
- Columns: `Image_name`, `Label`, `Source`. No patient identifier or narrative EMR is provided in this table.

## Binary-segmentation audit
- Eligible: 280 images (132 healthy, 118 inactive, 30 active).
- Excluded from this task: 132 nonhealthy images without a matching mask.
  They remain in the original archive and are not silently assigned negative masks.
- Exact decoded-pixel duplicates: 0.
- 626 coarse dHash pairs were flagged; none passed the additional RGB RMSE test.
  Visually reviewed coarse matches included clearly different fundus images.
- 173 mask-read warnings concern nonbinary JPEG intensity values. Masks are
  thresholded at 127 (0 for genuinely binary 0/1 masks); original bytes are retained.
- Frozen binary-v2 exploratory split: 196 training, 42 validation, 42 test images.

## Limits
No verified patient linkage, paired narrative EMR or longitudinal outcome.
Image-level splitting is exploratory; hash checks cannot prove patient independence.
Active/inactive masks are unioned for segmentation, not used to predict activity.
Annotation completeness limits the eligible cohort and may introduce selection bias.
A small visual QA panel is not a full expert review of all annotations.

Status: archive verified; machine audit and sampled visual QA complete for the
binary pilot. Clinical annotation adjudication and patient-linkage verification remain open.
