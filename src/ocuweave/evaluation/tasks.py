"""Task-specific original-grid validation, bound to the training manifest."""

from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from ocuweave.data.tasks import SegmentationTask
from ocuweave.inference.tasks import load_checkpoint, image_probabilities
from ocuweave.evaluation.segmentation import binary_metrics, aggregate
from ocuweave.utils.provenance import sha256, write_json


def evaluate_task(
    task, manifest, source, checkpoint, output, *, device="cpu", allow_unverified=False
):
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve previous evaluations")
    data = SegmentationTask(task, manifest, source, allow_unverified=allow_unverified)
    model, state, stored_task, config, classes = load_checkpoint(checkpoint, device)
    recorded = state.get(
        "manifest_sha256",
        state.get("protocol", {}).get(
            "manifest_sha256", state.get("provenance", {}).get("manifest_sha256")
        ),
    )
    if task != stored_task or recorded != data.manifest_sha256:
        raise ValueError("Checkpoint task/manifest mismatch")
    if state.get("format") == "ocuweave-task-v1" and not state.get("complete"):
        raise ValueError("Interrupted run is not eligible for final validation")
    output.mkdir(parents=True, exist_ok=False)
    dev = next(model.parameters()).device
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    started = time.monotonic()
    rows = {c: [] for c in classes}
    for row in data.manifest["records"]:
        if row["split"] != "validation":
            continue
        image, targets, available = data.original(row)
        p = image_probabilities(model, image, config["size"], dev)
        for c, name in enumerate(classes):
            predicted = p[c] >= 0.5
            Image.fromarray(predicted.astype(np.uint8) * 255).save(
                output / (row["sample_id"] + "-" + name + ".png")
            )
            if not available[c]:
                continue
            if task == "fives":
                from ocuweave.training.vessels import vessel_metrics

                metrics = vessel_metrics(predicted, targets[c])
            else:
                metrics = binary_metrics(predicted, targets[c])
            metrics["false_positive_area_fraction"] = metrics["fp"] / targets[c].size
            rows[name].append({"sample_id": row["sample_id"], **metrics})
    summaries = {c: aggregate(items) if items else None for c, items in rows.items()}
    for c, items in rows.items():
        if not items:
            continue
        for key in (
            "false_positive_area_fraction",
            "cldice",
            "thin_centerline_recall",
            "sensitivity",
            "precision",
        ):
            values = [r[key] for r in items if r.get(key) is not None]
            if values:
                summaries[c][key] = float(np.mean(values))
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    result = {
        "task": task,
        "split": "validation",
        "grid": "original",
        "threshold": 0.5,
        "test_evaluated": False,
        "manifest_sha256": data.manifest_sha256,
        "checkpoint_sha256": sha256(checkpoint),
        "per_class": summaries,
        "per_image_class": rows,
        "elapsed_seconds": time.monotonic() - started,
        "patient_identity_verified": data.manifest.get(
            "patient_identity_verified", False
        ),
    }
    write_json(output / "metrics.json", result)
    return result
