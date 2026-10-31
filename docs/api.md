# Python API and extension boundaries

## Image-only models

```python
import torch
from ocuweave.models.local_detail import LocalDetailSegmenter
from ocuweave.losses.segmentation import segmentation_loss

model = LocalDetailSegmenter(width=24, dim=48, queries=8)
images = torch.rand(2, 3, 192, 192)  # RGB float intensities in [0, 1]
reference = torch.zeros(2, 1, 192, 192)
valid = torch.ones_like(reference)  # Exclude actual letterbox padding in real batches
logits = model(images)             # B × 1 × H × W; no sigmoid inside the model
loss = segmentation_loss(logits, reference, valid)
loss.backward()
```

This is a shape/gradient example, not a training dataset. Use `OcuWeaveSegmenter` from `ocuweave.models.segmenter` for the query-only baseline. `make_model(config)` in `ocuweave.models.factory` resolves `query`, `query_detail`, and `fcn`. Inputs require RGB channels and spatial dimensions at least 16; protocol configurations use multiples of 16. Output is a single binary foreground logit map.

## Training and prediction

```python
from ocuweave.training.engine import train
from ocuweave.inference.segmentation import predict_image
from PIL import Image

# `config` must contain the exact fields emitted by `ocuweave init-config`.
checkpoint = train(manifest, archive, output_directory, config,
                   allow_unverified=False, device="cpu")
with Image.open(image_path) as image:
    probability = predict_image(checkpoint, image, device="cpu")
# NumPy float array (original height, original width), probabilities in [0, 1].
```

Use `save_prediction(checkpoint, image_path, output, device="cpu")` to save the same artifacts as the CLI. Inference and original-grid evaluation share `restore_probabilities`, so padding removal/interpolation stay consistent. Standard PyTorch state dictionaries are saved with config and provenance; no TorchScript, ONNX or C++ inference export is provided.

## Vessel and four-channel lesion workflows

`prepare_fives` / `prepare_idrid` in `ocuweave.data.adapters` parse extracted releases into immutable manifests. Use `scripts/run_fives.py` and `scripts/run_idrid.py` for the separately specified preparation and three-variant feasibility studies. They preserve official test partitions and evaluate only the development validation split. Read the task protocols before enabling provisional grouping.

For IDRiD, `MultiLabelLQNet` in `ocuweave.models.multilabel` returns **B × 4 × H × W** logits, ordered MA, HE, EX, SE. `observed_loss` in `ocuweave.training.lesions` accepts a **B × 4** boolean annotation-availability mask; absent annotations do not contribute gradients.

```python
from ocuweave.inference.multilabel import predict_lesions, save_lesion_prediction
from PIL import Image

with Image.open("fundus.jpg") as image:
    probabilities = predict_lesions("final.pt", image, device="cpu")
# Float NumPy array: 4 × original height × original width.

save_lesion_prediction("final.pt", "fundus.jpg", "outputs/lesion-prediction")
# MA.png, HE.png, EX.png, SE.png and prediction.json; threshold 0.5.
```

The IDRiD checkpoint schema is distinct from the generic binary CLI. Use `scripts/predict_idrid.py` or the API above; an incompatible binary checkpoint is rejected. `save_probabilities=True` additionally writes the original-grid float array as `probabilities.npy`. It is omitted by default. Empty output masks are valid predictions, not evidence that a photograph is disease-free.

## Extending the project

A new image-only architecture should implement `forward(images) -> logits` with the same shape contract, register explicitly in the factory, validate its config and include gradient/checkpoint tests. The OTFID/FIVES paths are binary; the IDRiD path has four independent lesion channels. Artery/vein or disc/cup segmentation still requires its own target schema, adapter and per-structure evaluation; changing the display label is insufficient.

Clinical, language and fusion packages remain design boundaries. Do not pass generated notes or unrelated patient records to the image-only model. Native `evidence_weighted_mean` is an optional numerical example without autograd; it is not the neural network or the proposed evidence-aware fusion method.

## Reproducibility scope

Seed/configuration, archive and manifest hashes, software version and available Git revision are recorded in training provenance. Installed wheels without checkout metadata report the revision as unavailable. Dirty source trees are explicitly marked. Report package versions and hardware alongside any comparison. The software is under development: pin a release tag for external replication.
