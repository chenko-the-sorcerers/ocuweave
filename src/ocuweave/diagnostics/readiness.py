"""Fixed-subset fitting, foreground probabilities, loss and parameter diagnostics."""

from pathlib import Path
import math
import time
import numpy as np
import torch
from PIL import Image
from ocuweave import __version__
from ocuweave.data.tasks import SegmentationTask
from ocuweave.evaluation.segmentation import binary_metrics
from ocuweave.inference.tasks import require_device, image_probabilities
from ocuweave.losses.observed import observed_components
from ocuweave.training.engine import seed_everything
from ocuweave.training.task_config import task_model, validate_config
from ocuweave.training.tasks import prepare_batch, atomic_checkpoint
from ocuweave.utils.provenance import sha256, write_json, software_provenance


def probability_stats(values):
    if values.numel() == 0:
        return None
    a = values.detach().float().cpu().numpy()
    return {
        "pixels": int(a.size),
        "mean": float(a.mean()),
        "min": float(a.min()),
        "max": float(a.max()),
        "p05": float(np.quantile(a, 0.05)),
        "median": float(np.median(a)),
        "p95": float(np.quantile(a, 0.95)),
        "fraction_at_or_above_0_5": float((a >= 0.5).mean()),
    }


def batch_observation(logits, batch, classes):
    out = []
    p = logits.sigmoid()
    for i, sid in enumerate(batch["id"]):
        for c, name in enumerate(classes):
            if not batch["available"][i, c]:
                out.append(
                    {
                        "sample_id": sid,
                        "class": name,
                        "annotation_available": False,
                        "metrics": None,
                    }
                )
                continue
            v = batch["valid"][i, 0].bool()
            y = batch["mask"][i, c].bool()
            prob = p[i, c]
            out.append(
                {
                    "sample_id": sid,
                    "class": name,
                    "annotation_available": True,
                    "reference_pixels": int((v & y).sum()),
                    "foreground_probability": probability_stats(prob[v & y]),
                    "background_probability": probability_stats(prob[v & ~y]),
                    "metrics": binary_metrics(
                        (prob >= 0.5).detach().cpu().numpy(),
                        y.cpu().numpy(),
                        v.cpu().numpy(),
                    ),
                }
            )
    return out


def parameter_group(name):
    if name.startswith("encoder."):
        return "encoder"
    if name.startswith("heads."):
        return ".".join(name.split(".")[:2])
    if name.startswith(("detail.", "refine.")):
        return "local_detail"
    return name.split(".")[0]


def diagnostics(
    task,
    manifest,
    source,
    output,
    config,
    *,
    device="cuda",
    allow_unverified=False,
    count=2,
    sample_ids=None,
):
    validate_config(config)
    if count not in (2, 3, 4):
        raise ValueError("Select two to four training images")
    dev = require_device(device)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new diagnostic directory")
    data = SegmentationTask(task, manifest, source, allow_unverified=allow_unverified)
    ds = data.dataset("train", config["size"])
    ordered = sorted(range(len(ds)), key=lambda i: ds.rows[i]["sample_id"])
    if sample_ids:
        if len(sample_ids) != count or len(set(sample_ids)) != count:
            raise ValueError("Unique sample IDs must match count")
        lookup = {r["sample_id"]: i for i, r in enumerate(ds.rows)}
        if any(s not in lookup for s in sample_ids):
            raise ValueError("Only training IDs may enter diagnostics")
        indices = [lookup[s] for s in sample_ids]
    else:
        indices = ordered[:count]
    if len(indices) != count:
        raise ValueError("Insufficient training samples")
    seed_everything(config["seed"])
    model = task_model(task, config).to(dev)
    batch = prepare_batch(ds, indices, dev)
    output.mkdir(parents=True, exist_ok=False)
    protocol = {
        **software_provenance(),
        "purpose": "training-only fixed-subset diagnostic; not research performance",
        "task": task,
        "sample_ids": batch["id"],
        "selection": (
            "explicit IDs" if sample_ids else "first training IDs in lexical order"
        ),
        "config": config,
        "augmentation": "disabled",
        "device": str(dev),
        "gpu_executed": dev.type == "cuda",
        "device_name": (
            torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
        ),
        "manifest_sha256": data.manifest_sha256,
        "precision": "float32",
        "test_evaluated": False,
        "validation_evaluated": False,
        "readiness": "manual review required; no automatic permission to start the main matrix",
    }
    write_json(output / "protocol.json", protocol)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    start = time.monotonic()
    history = []
    try:
        model.eval()
        with torch.no_grad():
            initial = batch_observation(model(batch["image"]), batch, data.classes)
        write_json(output / "initial.json", initial)
        for step in range(1, config["steps"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            values = observed_components(
                model(batch["image"]), batch["mask"], batch["valid"], batch["available"]
            )
            values["total"].backward()
            groups = {}
            before = {name: p.detach().clone() for name, p in model.named_parameters()}
            for name, p in model.named_parameters():
                group = groups.setdefault(
                    parameter_group(name),
                    {
                        "gradient_squared": 0.0,
                        "update_squared": 0.0,
                        "without_gradient": 0,
                    },
                )
                if p.grad is None:
                    group["without_gradient"] += 1
                    continue
                if not torch.isfinite(p.grad).all():
                    raise FloatingPointError("Nonfinite gradient")
                group["gradient_squared"] += float(p.grad.detach().square().sum())
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            optimizer.step()
            for name, p in model.named_parameters():
                groups[parameter_group(name)]["update_squared"] += float(
                    (p.detach() - before[name]).square().sum()
                )
            record = {
                "step": step,
                **{k: float(v.detach()) for k, v in values.items()},
                "gradient_norm": float(norm),
                "parameter_groups": {
                    k: {
                        "gradient_l2": math.sqrt(v["gradient_squared"]),
                        "update_l2": math.sqrt(v["update_squared"]),
                        "parameters_without_gradient": v["without_gradient"],
                    }
                    for k, v in groups.items()
                },
            }
            history.append(record)
            if step % config["save_every"] == 0 or step == config["steps"]:
                write_json(output / "history.json", history)
        model.eval()
        with torch.no_grad():
            expected = model(batch["image"])
            final = batch_observation(expected, batch, data.classes)
        write_json(output / "final-input-grid.json", final)
        checkpoint = output / "diagnostic.pt"
        atomic_checkpoint(
            checkpoint,
            {
                "format": "ocuweave-task-v1",
                "software_version": __version__,
                "task": task,
                "classes": list(data.classes),
                "config": config,
                "model": model.state_dict(),
                "manifest_sha256": data.manifest_sha256,
                "complete": False,
                "diagnostic": True,
                "step": config["steps"],
                "provenance": protocol,
            },
        )
        restored = task_model(task, config).to(dev)
        restored.load_state_dict(
            torch.load(checkpoint, map_location="cpu", weights_only=True)["model"]
        )
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(expected, restored(batch["image"]))
        original = []
        for i in indices:
            row = ds.rows[i]
            image, targets, available = data.original(row)
            p = image_probabilities(model, image, config["size"], dev)
            sample_dir = output / row["sample_id"]
            sample_dir.mkdir()
            display = image.copy()
            display.thumbnail((800, 800))
            display.save(sample_dir / "photo.png")
            for c, name in enumerate(data.classes):
                metric = (
                    binary_metrics(p[c] >= 0.5, targets[c]) if available[c] else None
                )
                original.append(
                    {
                        "sample_id": row["sample_id"],
                        "class": name,
                        "annotation_available": bool(available[c]),
                        "reference_pixels": (
                            int(targets[c].sum()) if available[c] else None
                        ),
                        "metrics": metric,
                    }
                )
                for label, mask in [
                    ("prediction", p[c] >= 0.5),
                    ("reference", targets[c] if available[c] else None),
                ]:
                    if mask is None:
                        continue
                    Image.fromarray(mask.astype(np.uint8) * 255).save(
                        sample_dir / (name + "-" + label + ".png")
                    )
        coverage = {
            c: any(
                r["class"] == c
                and r.get("reference_pixels", 0)
                and r["annotation_available"]
                for r in final
            )
            for c in data.classes
        }
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        report = {
            **protocol,
            "status": "executed_requires_review",
            "checkpoint_reload_matches": True,
            "checkpoint_sha256": sha256(checkpoint),
            "positive_supervision_on_input_grid": coverage,
            "untested_classes": [c for c, ok in coverage.items() if not ok],
            "initial": initial,
            "final": final,
            "original_grid": original,
            "steps_completed": len(history),
            "first_loss": history[0]["total"],
            "last_loss": history[-1]["total"],
            "elapsed_seconds": time.monotonic() - start,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
            ),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None
            ),
        }
        write_json(output / "diagnostics.json", report)
        return report
    except Exception as error:
        write_json(
            output / "failed.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "steps_completed": len(history),
                "test_evaluated": False,
            },
        )
        raise
