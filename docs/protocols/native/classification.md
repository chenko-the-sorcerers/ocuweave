# Native disease classification and Kaggle

## Scope

The classifier shares the independently implemented LQNet residual encoder. Global-average pooling of its deepest features feeds a linear four-class head. Every parameter starts from random initialization. Training uses stable multiclass cross entropy, its analytic derivative and the native AdamW implementation. This is an image-level disease classification experiment; localization remains the separate LQNet segmentation task.

FIVES filename annotations identify A = AMD, D = diabetic retinopathy, G = glaucoma and N = normal, as described by [Jin et al., Scientific Data (2022)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9352679/). Neither filenames nor vessel masks enter the network. These categorical labels do not establish disease onset, severity, or future vision loss.

## Frozen experiment

`configs/native/classification-fives.json` registers seed 42, 192-pixel bilinear letterboxing, encoder width 24, batch size 8, 1,200 optimizer steps, learning rate 0.001, weight decay 0.0001 and gradient clipping at 1. No augmentation is applied in this initial baseline. The existing manifest SHA-256 is required; do not regenerate a more favorable split. Similarity groups are provisional and patient identity remains unverified. Official test images are excluded.

Final validation accuracy >0.85 is a target, not an early-stopping rule or current result. Report balanced accuracy, macro F1, class counts, per-class sensitivity/specificity and one-vs-rest AUROC alongside accuracy. The 0.5 normal-versus-abnormal probability threshold is fixed and exploratory; it is not a clinical screening threshold. Segmentation validation Dice >0.80 is a separate target evaluated at original image size and per annotated target.

## Reproduction

Build from a clean release checkout with CMake and `OCUWEAVE_ENABLE_CUDA=ON`, specifying the architecture of the actual allocated GPU. Run CTest, including `native_classification_cuda`, before medical training. Python orchestration needs Pillow, not an ML framework.

```bash
python scripts/native_classification_programme.py \
  --repo "$PWD" --build /path/to/build \
  --root /path/to/FIVES --manifest /path/to/frozen/manifest.json \
  --session /path/to/new-session --expected-commit "$(git rev-parse HEAD)"
```

Use `--wait-session /path/to/native-segmentation-session` only when sharing the same GPU with its core and patch jobs. A separate Kaggle GPU can run classification independently without changing those jobs. The runner refuses existing session directories. Outputs include the contract, source/binary/data hashes, training logs, initial/latest checkpoints, predictions, confusion matrices, independent metric recount and completion/failure status.

Native CLI `--resume` restores model and AdamW state; `--steps` means additional steps. Deterministic batching continues from the optimizer step. Before manual continuation, verify the same data hashes, seed, width, class order, batch size and learning rate. The low-level checkpoint does not itself bind every data/CLI setting; never describe unchecked resumes as exact reproductions. Segmentation and classification checkpoints have different parameter sets and are not interchangeable.

## Kaggle

Import `notebooks/OcuWeave_Native_Kaggle.ipynb` into a private Kaggle notebook. Enable internet and a GPU after completing Kaggle's account verification. The notebook resolves the release tag to a recorded commit and refuses a different existing checkout. It does not mount Google Drive or embed credentials.

Attach the frozen FIVES manifest and original dataset as private inputs; set the two explicit paths in the input cell. Paths are relative to the extraction root recorded by the manifest, and every used image is checksum verified. Do not substitute a Kaggle mirror without verifying source provenance and bytes. Keep the original CC BY 4.0 attribution/license. The default run flag is false so importing the notebook cannot start an expensive run with guessed inputs.

Save a notebook version and preserve `/kaggle/working/ocuweave-runs` outputs before ending a session. Kaggle outputs are not automatically copied to Drive. Download or transfer them to the project's Drive runs folder, including checkpoints, hashes and logs. Existing Colab checkpoints must be inspected for the last complete saved step before any segmentation migration; this launcher does not automatically resume that matrix.

## Status on 2026-11-01

CPU synthetic tests are available and exercised. Kaggle GPU and internet are currently locked pending phone verification, which the user cannot complete yet. Colab reports a GPU usage limit. New classification CUDA parity and medical training are unexecuted; previously recorded segmentation evidence remains unchanged. Hardware changes must be recorded when computation resumes.
