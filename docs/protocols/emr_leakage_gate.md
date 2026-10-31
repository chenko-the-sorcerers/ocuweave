# Entry criteria for original-note multimodal glaucoma research

Author: Marchel Andrian Shevchenko. Date: 2026-10-31. Status: **not cleared for training**.

FairVLMed is a separate paired SLO/clinical-note glaucoma cohort. OTFID is the
image-only toxoplasmosis segmentation cohort. Never combine patients across these
sources to manufacture image–EMR pairs. Generated summaries or metadata-derived
sentences are not original EMR and must not be described as such.

## Review before model fitting

Define the prediction time and clinical information available at that time.
Review training/validation original notes for diagnosis assertions, negation scope,
suspected/risk language, family history, laterality, historical vs current events,
postdiagnosis treatment, visual-field/RNFL measurements and other endpoint proxies.
The existing lexical flags are a review queue, not semantic adjudication.

Use the review form in `docs/results/emr-review-template.csv` inside the authorized
private Drive project. Record only filename, split, evidence category, reviewer,
review date, timing status, rationale without verbatim note text, and action.
Do not publish original notes or identifiable snippets. A clinician must assess
ambiguous clinical interpretation; no such collaborator or completed review is
currently claimed. Unknown temporal ordering remains unknown, not presumed safe.

A stratified sample can identify failure patterns, but does not certify the entire
corpus. Audit any proposed deterministic sanitization on held-out development
notes and document residual-risk examples privately. Removing the literal word
“glaucoma” alone does not remove proxies. Never label the unflagged subset safe
without supporting review. Preserve raw originals and transformation provenance.

## Training gate and experiments

The gate opens only after the task definition, timing assumptions, allowed feature
schema, sanitization policy, duplicate/template handling, patient-linkage limits,
license/access scope, reviewer decisions and residual risks are recorded. Current
status is pending. Do not inspect test-note content to design the policy.

Then predeclare image-only, reviewed-original-note-only and paired multimodal
baselines using the official split, explicit missingness masks and comparable
budgets. Exclude diagnosis label, split/filename identifiers, GPT-generated
summaries, and outcome measurements from model inputs unless a separately defined
clinical task justifies them. A note-only baseline is a leakage probe as well as
a comparator; high performance does not alone demonstrate useful visual grounding.
Evaluate missing-note robustness, calibration, subgroup coverage and source/label
consistency. Preserve final test access for a frozen policy/model comparison.

No toxoplasmosis EMR, follow-up interval, glaucoma progression time, or future
vision-loss outcome is inferred from this cross-sectional dataset preparation.
