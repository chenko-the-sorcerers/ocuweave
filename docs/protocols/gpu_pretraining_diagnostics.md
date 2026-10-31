# GPU readiness checklist before further research training

Marchel Andrian Shevchenko · 1 November 2026

**Status: planned, not executed.** Record this checkpoint for the next authorized GPU session. Do not start the full experiment matrix merely because a GPU becomes available. The small-set fitting below is diagnostic training; it precedes the main research runs. This document does not schedule a job or allocate compute.

## Evidence motivating the checks

The [v0.10.0 feasibility report](../results/cross-task-feasibility-v0100/README.md) records six 100-step CPU runs. FIVES LQNet image-macro Dice is 0.2022, vessel recall approximately 0.1500 and thin-centerline recall 0.0540. The local branch does not improve thin-centerline recall over Query 384. Every annotated IDRiD validation channel has TP = 0 and FP = 0 at threshold 0.5, despite finite gradients and decreasing training loss. These are measured failures; their causes are not yet isolated. A faster device alone is not evidence that the failure is resolved.

## Ordered checks

- [ ] **Freeze the diagnostic contract.** Record GPU model, software/environment, source commit, data/manifest hashes, precision, seed, input size, step budget, optimizer, learning rate, checkpoint policy and acceptance criteria before execution. Keep diagnostic run IDs separate from the core programme. Preserve the historical OTFID source and manifest contract.
- [ ] **Inspect the actual tensors sent to the GPU.** Review photo/mask alignment, native label decoding, channel order, positive-pixel counts before/after resizing, padding exclusion and annotation availability. Retain flagged images and all original labels. IDRiD missing masks remain unknown supervision.
- [ ] **Run CUDA smoke checks.** Verify finite forward outputs, BCE and soft-Dice components, nonzero finite gradients, optimizer updates, memory use and checkpoint reload agreement under declared tolerances. Start with float32; any mixed-precision check is a separate comparison.
- [ ] **Fit a fixed set of 2–4 training images.** Select deterministically from training only and record IDs before fitting; cover the observed lesion channels where possible. Disable stochastic augmentation for this diagnostic. Lock the budget and criteria first. Inspect predicted foreground, true-positive pixels and Dice/IoU per image/class on both the network grid and the restored original grid. If a class has no positive supervision in this subset, mark its check untested. A falling aggregate loss alone does not pass this gate.
- [ ] **Diagnose background predictions.** Log foreground/background probability distributions, predicted-positive fractions at the fixed 0.5 threshold, separate BCE/Dice components, annotation counts and gradient/update magnitudes for the encoder, each query head and the local branch. Inspect training data first. Do not select a threshold from the public gallery or the test set. Preserve failed runs.
- [ ] **Register a full-image versus patch/high-resolution experiment.** Audit small-structure survival, define crop coordinates and sampling using training data only, and test full-image reconstruction and overlap/padding handling. Split/group source images before creating patches. Lock effective batch, optimization budget, sampled-pixel exposure and inference coverage. Report unavoidable compute differences; do not silently substitute patches into the original 192/384 protocol.
- [ ] **Evaluate loss changes separately.** If the diagnostics warrant foreground/class weighting or another objective, derive statistics from training only and predeclare coefficients and tuning budget. Hold architecture and sampling constant for this comparison. Do not change resolution, sampling and loss simultaneously and attribute the outcome to one component.
- [ ] **Review readiness and freeze the main configuration.** Record pass/fail/untested for each check with artifact links. Explain remaining small-set fitting errors, probability behavior and geometry limitations. Proceed to the main matrix only after unresolved pipeline failures are addressed and the evaluation contract is frozen. Diagnostic fitting success is not generalization or clinical validation.

## Evidence to save in Drive

For each diagnostic, retain configuration and hashes, selected training IDs, label-availability flags, tensor/overlay examples, loss components, gradient/update summaries, foreground probabilities, per-image/class metrics, time, peak GPU allocated/reserved memory, checkpoint hash and reload receipt. Save a readiness decision explaining failures and unresolved checks. Missing measurements remain null. Do not replace the six existing CPU result records.

## Boundaries

The diagnostic subset is training data only. Validation supports later development under its declared budget; old test results and protected test assignments remain unchanged. FIVES cross-official-partition similarity candidates still bar official-test claims until adjudicated; patient identity is unverified. REFUGE2 requires legitimate dataset access before its task can be implemented or trained. FairVLMed/EMR remains a separate glaucoma track pending leakage and timing review.

After the gate, follow the [cross-task programme](cross_task_experiments.md) and the [original OTFID continuation protocol](exploratory_training.md). Any protocol amendment receives a new configuration/version; it must not overwrite frozen runs or masquerade as completion of the earlier design.
