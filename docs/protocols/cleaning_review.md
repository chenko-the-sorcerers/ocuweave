# Cleaning review: quality, annotation, identity and original EMR

This protocol creates review priorities and auditable decision forms. It neither deletes records nor certifies clinical correctness. Run each cohort independently. No OTFID–FairVLMed patient join is permitted.

## Image quality

Measure grayscale Laplacian variance (sharpness), supported mean intensity and 95th–5th percentile contrast, bright clipping, dark area, non-dark border contact, and quadrant intensity range on a 256-pixel proxy. Non-dark support is not retinal-field segmentation. SLO borders and color fundus borders have different acquisition geometry.

Fit 10th/90th percentile cutoffs on training images only, separately per cohort. Flag low sharpness/intensity/contrast and high clipping/dark area/asymmetry. Border contact must exceed both 0.1 and its training 90th percentile. Strict inequalities avoid flagging every tied value. These are relative outliers, not validated clinical rejection thresholds. Lesions and normal anatomy can also trigger these flags.

Export every development record, including unflagged controls. Display four high-flag-count examples and two low-flag-count controls per cohort, selected deterministically. These selected previews are not a complete clinical review or an unbiased estimate of defect prevalence. Record human findings as blur, illumination, artifact, truncation, uncertain, or none, with reviewer, date and rationale.

**Exclusion rule:** no exclusion is authorized by a numerical flag. A subsequent versioned decision manifest must identify unreadable/corrupt inputs, unresolved pairing/geometry errors, or clinician-adjudicated ungradable images under a prespecified task-specific criterion. Retain originals and report all exclusion counts/reasons by split. Do not exclude an image merely because the model performs poorly. Any mask revision requires clinical approval, provenance and a new annotation version.

## Annotation review

Use the existing original/192/384 overlay audit and a queue covering all 238 development records. Verify lesion location, completeness, boundary and whether the union of activity masks matches the binary task. Healthy empty masks are explicit review items. Technical shape/foreground agreement cannot establish clinical correctness. An ophthalmologist must adjudicate; every initial row is `unreviewed`, with blank reviewer/date/rationale. No automated label correction is performed.

## Patient linkage and near duplicates

Retain acquisition receipts and exported column names. Distinguish source-reported patient counts from a verified image-to-patient/eye/visit mapping. Filenames, near duplicates, demographic matches and identical notes do not establish identity. Obtain a de-identified mapping with its definition and provenance from the custodian; no custodian has been contacted by this audit.

For integrity screening only, compute image descriptors for all frozen/official splits. Normalize central grayscale 64×64 patches, screen with 16×16 cosine similarity ≥0.995, then require normalized 64×64 RMSE ≤0.10. Check identity, horizontal flip and 180° rotation. Record cross-split candidate pairs, transform, similarity and review status. Limit to 20,000 candidates and explicitly report truncation. Low-variance descriptors are excluded and counted. Other crops/rotations, changes between visits and distinct views may be missed; zero candidates does not prove patient independence.

Test images are read only for numerical integrity fingerprints, with no test previews, masks, model evaluation or quality-threshold fitting. Test notes are parsed as part of the source CSV file but are skipped before semantic processing; no test-note text is exported or analyzed. Keep the frozen split and historical test results unchanged. Cross-split findings block independence claims pending investigation; do not use test performance to choose a remedy. Any future regrouping requires a new versioned protocol and separate evaluation.

## FairVLMed original notes

Analyze training and validation original notes only, separately from OTFID. For each diagnosis term, record character offsets and nearby negation, family/experiencer, suspicion, history and laterality cues within a clause, with a six-token left and three-token right window. Record treatment/endpoint cues and date-token presence at note level. No raw text or quoted snippets are exported.

These lexical flags are not a validated assertion, temporal or leakage resolver. Abbreviations and mixed clauses may be misread. A date token does not establish when a note was available. Human review must resolve assertion, experiencer, laterality, original encounter/prediction time, pre-index availability, treatment and outcome leakage. Unflagged notes are not cleared. No note is redacted, synthesized, deleted, paired across cohorts or admitted to training by this protocol.

## Required evidence and exit gates

Preserve hashes of raw archives, source CSV, frozen manifest, checkpoints and historical test JSON. Save code revision, CPU tests, quality summaries, per-record queues, duplicate pairs, linkage evidence, previews and an executed notebook in the private project Drive. Public documentation contains aggregate results and protocol only.

Cleaning is not complete until image-quality decisions, clinical annotation adjudication, patient-linkage verification and EMR semantic/temporal decisions are resolved for their respective tasks. Inability to obtain patient linkage must remain an explicit limitation, not be converted into an independence claim.

## Official sources

[1] R. Parra et al., “Ocular Toxoplasmosis Fundus Images Dataset (OTFID),” Zenodo, version 3, 2024. [Dataset record](https://zenodo.org/records/13883485).

[2] Harvard AI Robotics, “FairVLMed,” official dataset card. The card reports 10,000 patients/samples, original notes and generated summaries; these statements are not a supplied patient mapping. [Dataset card](https://huggingface.co/datasets/harvardairobotics/FairVLMed/blob/main/README.md).

[3] Y. Luo et al., “FairCLIP: Harnessing Fairness in Vision-Language Learning,” in Proc. IEEE/CVF CVPR, 2024. [Official paper](https://openaccess.thecvf.com/content/CVPR2024/papers/Luo_FairCLIP_Harnessing_Fairness_in_Vision-Language_Learning_CVPR_2024_paper.pdf).
