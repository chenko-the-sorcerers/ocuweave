# Architecture proposal

Images -> visual encoder -> global/region features.
Original notes -> local language encoder/LLM -> traceable text evidence.
Clinical variables -> structured encoder -> values and missingness.
These feed the proposed evidence fusion module and task-specific heads.

Only schema, manifest auditing, a model Protocol, and the CPU interop example are
implemented. A clinical note can contain information not visible in a fundus image;
absence of visual confirmation is not automatically a contradiction.

## Boundaries

- Data adapters own parsing. Models never open CSV files or select patient splits.
- ModelInputs excludes ground-truth labels and lesion masks.
- Labels/masks are supervision, passed separately to future training objectives.
- Python owns experiment composition; native owns isolated numerical operators.
- The C ABI catches exceptions. Python does not silently select a different backend.
- Native mean is a scalar CPU smoke-test operator with no autograd support.
- Pretrained encoders and their licenses will be chosen separately. Writing the
  method from scratch does not imply training an entire LLM from random weights.

## Reproducibility

Record code revision, configuration, dependency versions, dataset version, patient
split, seed, checkpoint source, and hardware. Select checkpoints on validation only.
