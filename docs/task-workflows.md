# Task workflows

Install the research extra from a pinned release as described in the README. The commands below work from an installed wheel outside the checkout. They use local authorized data; no downloads, account access or private uploads occur implicitly.

## Check the installation

```bash
ocuweave doctor
ocuweave demo --output outputs/binary-demo --device cpu
ocuweave demo-tasks --output outputs/task-demo --device cpu
ocuweave demo-multimodal --output outputs/evidence-demo
```

The demos generate geometric fixtures. Synthetic scores verify software plumbing, not retinal segmentation. The task demo covers FIVES/IDRiD formats, missing labels, train-only diagnostics and validation. Evidence conditioning is only a synthetic forward/backward check.

## Prepare data

```bash
ocuweave prepare-fives --root /data/FIVES --manifest /data/fives-development.json
ocuweave prepare-idrid --root /data/IDRiD --manifest /data/idrid-development.json
ocuweave prepare-otfid --archive /data/otfid.zip --output /data/otfid-audit \
  --allow-unverified-patients
```

Use an already frozen development manifest when reproducing an existing protocol. Preparation creates a new manifest; do not casually regenerate a published split. For FIVES, `--provisional-development` is an explicit opt-in when similarity candidates cross official partitions; review the audit first. It does not authorize test performance claims. Data quality flags are not exclusion decisions.

The shared `--source` argument is the **ZIP archive for OTFID** and the **extracted root directory for FIVES/IDRiD**. File/archive integrity is verified before training. Unknown IDRiD masks are excluded rather than treated as healthy. Unverified patients require `--allow-unverified-patients`; this permits exploratory grouping only.

## Diagnose before main training

```bash
ocuweave task-config --output outputs/diagnostic-config.json --size 384 --steps 50
ocuweave diagnose --task idrid --manifest /data/idrid-development.json \
  --source /data/IDRiD --config outputs/diagnostic-config.json \
  --output outputs/idrid-diagnostic-01 --device cuda \
  --count 4 --allow-unverified-patients
```

This is a fixed-subset training diagnostic, not the main experiment. It uses two to four **training** images, no augmentation, and an explicitly recorded budget. Use `--sample-ids ID1 ID2 ...` to choose training examples after reviewing class coverage. The default lexical selection may omit a class; the report lists untested classes. A successful execution is `executed_requires_review`, never automatic GPU readiness approval.

Inspect `diagnostics.json`, `history.json`, selected image/reference/prediction artifacts and `diagnostic.pt`. Records include foreground/background probability distributions, BCE and soft Dice, gradients and update norms by model component, missing annotations, original-grid overlap, time, CUDA memory and checkpoint reload checks. The checkpoint is marked diagnostic and cannot be used as a completed run by `score`. CPU use requires `--device cpu` and records `gpu_executed=false`.

The [readiness protocol](protocols/gpu_pretraining_diagnostics.md) still governs human review. Patch-based training, alternate objectives and clinical annotation correctness are not established by this command.

## Fixed-step training and resume

```bash
ocuweave task-config --output outputs/lqnet384-seed42.json \
  --architecture query_detail --size 384 --steps 500 --seed 42
ocuweave fit --task fives --manifest /data/fives-development.json \
  --source /data/FIVES --config outputs/lqnet384-seed42.json \
  --output outputs/fives-session-01 --device cuda \
  --max-steps-this-session 100 --allow-unverified-patients
ocuweave fit --task fives --manifest /data/fives-development.json \
  --source /data/FIVES --config outputs/lqnet384-seed42.json \
  --output outputs/fives-session-02 --device cuda \
  --resume outputs/fives-session-01/last.pt --allow-unverified-patients
```

The total budget remains 500 steps; the first command stops after 100 and the second completes the remaining 400. Periodic atomic checkpoints retain model, optimizer, scheduler, CPU/CUDA RNG and history. Resume requires identical task, manifest, configuration, software version and device type. Keep the same code revision, dependencies and hardware as well; exact CPU continuation is tested, but cross-hardware or GPU bitwise equality is not promised.

Outputs include `last.pt`, `history.json`, `provenance.json`, and `completed.json` or `interrupted.json`; exceptions write `failed.json` when the run has started. New output directories prevent accidental overwrite. The final fixed step is used, with no validation-selected checkpoint. This is protocol `fixed-step-task-v1`, distinct from earlier epoch-based and feasibility protocols. Preserve their results and configurations.

## Validate and predict

```bash
ocuweave score --task fives --manifest /data/fives-development.json \
  --source /data/FIVES --checkpoint outputs/fives-session-02/last.pt \
  --output outputs/fives-validation --device cuda --allow-unverified-patients
ocuweave infer --checkpoint outputs/fives-session-02/last.pt \
  --image /data/example.png --output outputs/prediction --device cpu
```

`score` checks the checkpoint's task and manifest hash and rejects interrupted task runs. It evaluates validation only, restores probabilities to original dimensions before threshold 0.5, excludes unknown annotation channels, and saves per-image/per-class overlap and false-positive area. FIVES also reports centerline metrics. Empty-reference and positive-only aggregates remain separate. It does not inspect the test split for model selection.

`infer` saves `probabilities.npy` (C×H×W), class masks and hashes in `prediction.json`. New task checkpoints and historical feasibility checkpoints are supported. Optional `--tile-size 512 --overlap 64` covers the original image, predicts each crop and averages overlapping probabilities before thresholding. Tiled inference changes the protocol and requires separate validation; it is not patch-based training or a demonstrated accuracy improvement.

## Implementation status

| Capability | Status |
|---|---|
| OTFID/FIVES/IDRiD preparation, training, validation, inference | Implemented; synthetic CPU integration tests |
| Interrupted training continuation | Implemented; exact CPU equivalence test |
| Training-only diagnostics | Implemented; synthetic CPU verified; initial medical-data T4 diagnostics executed; small-set fitting goal not passed |
| Note, structured feature and missing-modality conditioning | Implemented prototype; synthetic tests only |
| Native confusion counts and pooling | C++/Python parity tested; no autograd or speed claim |
| REFUGE2 training | Pending authorized data and adapter validation |
| Real EMR multimodal training and longitudinal prognosis | Pending data pairing, leakage review and research protocol |
| Patch training / objective comparisons / core GPU matrix | Pending registered execution |

The legacy `train/evaluate/predict` commands remain available for binary experiments. `scripts/evaluate.py` now forwards to the validation-first installed CLI; explicitly request `--grid input --split test` only when reproducing an authorized historical test protocol. Neither `score` nor diagnostics exposes a test-split option.
