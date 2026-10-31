# Quality review after the first segmentation pilot

Protocol date: 2026-10-31. Software v0.3.0. Author: Marchel Andrian Shevchenko.

## Decision

Determine whether the acquired arrays are technically usable and identify the
checks still needed before multimodal training. Diagnose the existing OTFID
checkpoint on development data, without choosing changes from the test results.
No training or model selection is performed by this protocol.

## FairVLMed

The [official dataset card](https://huggingface.co/datasets/harvardairobotics/FairVLMed)
describes 10,000 patients and SLO fundus images, original de-identified notes and
generated summaries. This is a source-reported patient count; the exported
identifiers have not independently established patient linkage in this project.
The source is a glaucoma cohort, not a toxoplasmosis cohort.

Read numeric `slo_fundus` and `glaucoma` arrays with `allow_pickle=False` after
checking primitive numeric types, dimensions and bounded allocation. Preserve the
original archive. Report corruption, nonfinite values, constant images, exact
pixel duplicates within/across official splits, and disagreement between the CSV
and NPZ label. Do not silently remove or relabel records. Exact hashes do not
identify all near duplicates. Image statistics do not establish clinical quality.

Study only training/validation original notes for lexical review flags. Keep
diagnosis terms, suspect/risk language, negation, family context, treatment terms,
and endpoint proxies as separate booleans. These are broad co-occurrences: they do
not resolve negation scope, temporal ordering, laterality, or clinical meaning.
Do not publish notes or summaries. Do not declare flagged or unflagged notes
safe for model input. Do not use `glaucoma`, visual-field `md`/`tds`, filenames, or
split identifiers as clinical input features. The latter endpoint fields are
excluded unless a separately justified study explicitly requires them.

Image integrity checks may cover all official splits, but test notes and model
performance remain outside the development analysis. Official splits remain
unchanged. The CC BY-NC-ND 4.0 dataset license is separate from the MIT software
license. No dataset derivative or trained weights are redistributed here.

## OTFID

Reuse the frozen binary-v2 manifest and the v0.2.0 seed-42 checkpoints. Compute
predictions on training and validation at the original 192-pixel input size and
fixed threshold 0.5. Preserve the two saved test evaluations. Summarize foreground
Dice by source, activity label, and original lesion-area bins: empty, >0–1%,
>1–5%, >5% of full image area. These bins are descriptive and are not optimized.
Report disappearing masks after resizing and healthy-image false-positive area.
Select the three lowest-Dice positive images and three healthy images with the
largest false-positive area for a deliberately adverse review panel. This panel
is not a representative sample or clinical diagnosis.

The validation set already selected the checkpoint, so these diagnostics do not
provide independent generalization estimates. Any new model should first be
developed on training/validation with a declared comparison and repeated seeds;
external or newly sealed evaluation is needed to support stronger claims.

## Artifacts

The notebook saves immutable JSON reports, selected failure figures, code revision,
artifact hashes and a validation log in the existing private Drive project.
Run the quality notebook on CPU if GPU quota is unavailable. It performs no
paid-service calls and no new model training.

Reference: Y. Luo et al., “FairCLIP: Harnessing Fairness in Vision-Language Learning,”
in *Proc. IEEE/CVF CVPR*, 2024, pp. 12289–12301.
[Official paper](https://openaccess.thecvf.com/content/CVPR2024/html/Luo_FairCLIP_Harnessing_Fairness_in_Vision-Language_Learning_CVPR_2024_paper.html).
