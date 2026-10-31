# FIVES — vessel segmentation cohort

Source: [Jin et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01564-3). The publication describes 800 fundus photographs from 573 patients and binary vessel masks, with 600 recommended training images and 200 test images. Disease strata cover normal, DR, glaucoma and AMD; these are image-level context, not lesion-mask classes or hypertension supervision.

The published archive layout uses train/test folders with Original and Ground Truth subfolders and matching PNG filenames. The official Figshare v1 archive was downloaded to project Drive and its MD5 verified on 2026-11-01; extraction found 800 image–mask pairs and the quality spreadsheet. The adapter validates pairing, dimensions, binary labels and file hashes. Patient identifiers must be inspected; filenames encoding disease and index do not establish patient independence.

Preserve the official test split. Construct validation only from training after reviewing available patient and duplicate-family links. Retain quality flags. Report unresolved patient linkage. The initial audit stopped on cross-official-partition similarity candidates. A documented provisional development mode preserves the official test and bars test-performance claims while grouping training members together. See the [feasibility protocol](../protocols/fives_feasibility.md); technical similarity groups do not establish patient identity.

The supplementary [100-step CPU feasibility report](../results/cross-task-feasibility-v0100/README.md) records Query 192, Query 384 and LQNet 384 validation results on 480 training / 120 validation images. All 200 official test images remain unevaluated.
