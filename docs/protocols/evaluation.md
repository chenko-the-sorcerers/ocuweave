# Proposed evaluation protocol

1. Verify patient/image/note linkage, permissions, duplicates, and laterality.
2. Separate diagnosis-time inputs from downstream assessment, treatment, and labels.
   Masking a disease keyword alone is insufficient to remove target leakage.
3. Split by patient, grouping both eyes and all visits. Detect duplicates before
   splitting. Never tune on held-out test data; preserve official splits when valid.
4. Compare image-only, text-only, structured features, late fusion, and OcuWeave.
5. Report disease-specific AUROC and AUPRC, operating-point sensitivity/specificity,
   calibration, and patient-bootstrap confidence intervals. For localization use
   lesion annotations and overlap metrics, not visual appeal of heatmaps alone.
6. Test absent notes, absent clinical fields, image-quality shifts, and controlled
   text/image inconsistency. Synthetic corruption results are not clinical validation.
7. Keep dataset endpoints separate: a BRSET scar label is not active toxoplasmosis;
   FairVLMed glaucoma and OLIVES DR/DME are distinct tasks. External validation
   requires compatible labels and independent patients.

No dataset in the current audit verifies complete paired EMR for active ocular
toxoplasmosis. OTFID patient grouping and class counts must be checked in the actual
archive. Metadata converted to sentences must remain labeled synthetic/serialized.
