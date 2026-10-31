# FIVES — candidate vessel segmentation cohort

Source: [Jin et al., Scientific Data, 2022](https://doi.org/10.1038/s41597-022-01564-3). The publication describes 800 fundus photographs from 573 patients and binary vessel masks, with 600 recommended training images and 200 test images. Disease strata cover normal, DR, glaucoma and AMD; these are image-level context, not lesion-mask classes or hypertension supervision.

The published archive layout uses train/test folders with Original and Ground Truth subfolders and matching PNG filenames. Verify the actual archive, license, provenance and quality-assessment file before implementing an adapter. Acquisition in the OcuWeave workspace is not yet verified. Patient identifiers must be inspected; filenames encoding disease and index do not establish patient independence.

Preserve the official test split. Construct validation only from training after reviewing available patient and duplicate-family links. Retain quality flags. Report unresolved patient linkage. All prospective experiments are described in [the protocol](../protocols/cross_task_experiments.md). No FIVES OcuWeave predictions or research metrics exist yet.
