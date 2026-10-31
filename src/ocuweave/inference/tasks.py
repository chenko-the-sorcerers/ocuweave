"""Task-aware checkpoint loading and full-image or tiled prediction."""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ocuweave.data.preprocessing.geometry import letterbox
from ocuweave.evaluation.original_grid import restore_probabilities
from ocuweave.training.engine import select_device
from ocuweave.training.task_config import task_model
from ocuweave.utils.provenance import sha256, write_json


def require_device(name):
    if name not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("Unsupported device")
    device = select_device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; choose CPU explicitly for software checks")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS unavailable")
    return device


def load_checkpoint(path, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    protocol = state.get("protocol", {})
    if state.get("format") == "ocuweave-task-v1":
        task = state["task"]
        config = state["config"]
        classes = state["classes"]
    elif protocol.get("protocol") == "idrid-feasibility-v1":
        task = "idrid"
        config = dict(
            architecture="query_detail" if state["local_detail"] else "query",
            width=24,
            dim=48,
            queries=8,
            size=state["size"],
        )
        classes = ["MA", "HE", "EX", "SE"]
    elif isinstance(state.get("config"), dict) and "model" in state:
        task = (
            "fives" if protocol.get("protocol") == "fives-feasibility-v1" else "otfid"
        )
        config = state["config"]
        classes = ["vessel" if task == "fives" else "lesion"]
    else:
        raise ValueError("Unsupported checkpoint format")
    expected = (
        ["MA", "HE", "EX", "SE"]
        if task == "idrid"
        else ["vessel" if task == "fives" else "lesion"]
    )
    if (
        classes != expected
        or type(config.get("size")) is not int
        or config["size"] < 16
    ):
        raise ValueError("Invalid checkpoint task contract")
    model = task_model(task, config).to(require_device(device))
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    return model, state, task, config, classes


@torch.no_grad()
def image_probabilities(model, image, size, device):
    image = image.convert("RGB")
    pixels, _, valid = letterbox(
        image, np.zeros((image.height, image.width), bool), size
    )
    x = (
        torch.from_numpy(pixels.transpose(2, 0, 1).copy()).float()[None].to(device)
        / 255
    )
    probability = model(x).sigmoid()
    return np.stack(
        [
            restore_probabilities(
                probability[:, c : c + 1], valid, (image.height, image.width)
            )
            .cpu()
            .numpy()
            for c in range(probability.shape[1])
        ]
    )


def infer(checkpoint, image_path, output, *, device="cpu", tile_size=None, overlap=0):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new output directory")
    model, state, task, config, classes = load_checkpoint(checkpoint, device)
    dev = next(model.parameters()).device
    with Image.open(image_path) as src:
        image = src.convert("RGB")
    if tile_size is None:
        if overlap:
            raise ValueError("Overlap requires --tile-size")
        p = image_probabilities(model, image, config["size"], dev)
        boxes = None
    else:
        from ocuweave.data.preprocessing.tiles import tile_boxes, merge_tiles

        boxes = tile_boxes(image.height, image.width, tile_size, overlap)
        tiles = [
            image_probabilities(model, image.crop((x, y, xx, yy)), config["size"], dev)
            for y, x, yy, xx in boxes
        ]
        p = merge_tiles(tiles, boxes, (image.height, image.width))
    output.mkdir(parents=True, exist_ok=False)
    np.save(output / "probabilities.npy", p)
    for c, name in enumerate(classes):
        Image.fromarray((p[c] >= 0.5).astype(np.uint8) * 255).save(
            output / (name + ".png")
        )
    receipt = {
        "task": task,
        "classes": classes,
        "shape": list(p.shape),
        "threshold": 0.5,
        "checkpoint_sha256": sha256(checkpoint),
        "input_sha256": sha256(image_path),
        "input_size": config["size"],
        "tile_size": tile_size,
        "overlap": overlap,
        "tile_count": len(boxes) if boxes else None,
        "clinical_use": False,
        "interpretation": "Research foreground masks; tiled inference changes the evaluation protocol and requires separate validation.",
    }
    write_json(output / "prediction.json", receipt)
    return receipt
