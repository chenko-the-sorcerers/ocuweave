"""Resumable fixed-step research runs, separate from historical frozen suites."""

import math
import os
import time
from pathlib import Path
import torch
from torch.utils.data import default_collate
from ocuweave import __version__
from ocuweave.data.tasks import SegmentationTask
from ocuweave.data.augmentation.paired import horizontal_flip
from ocuweave.inference.tasks import require_device
from ocuweave.losses.observed import observed_components
from ocuweave.training.engine import seed_everything
from ocuweave.training.task_config import validate_config, task_model
from ocuweave.utils.provenance import sha256, write_json, software_provenance


def atomic_checkpoint(path, state):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)


def step_indices(length, batch_size, seed, step):
    batches = math.ceil(length / batch_size)
    epoch, offset = divmod(step, batches)
    order = torch.randperm(
        length, generator=torch.Generator().manual_seed(seed + epoch)
    ).tolist()
    return order[offset * batch_size : (offset + 1) * batch_size]


def prepare_batch(dataset, indices, device):
    batch = default_collate([dataset[i] for i in indices])
    if "available" not in batch:
        batch["available"] = torch.ones(batch["mask"].shape[:2], dtype=torch.bool)
    for k in ("image", "mask", "valid", "available"):
        batch[k] = batch[k].to(device)
    return batch


def fit(
    task,
    manifest,
    source,
    output,
    config,
    *,
    device="cuda",
    allow_unverified=False,
    resume=None,
    max_steps_this_session=None,
):
    validate_config(config)
    dev = require_device(device)
    if max_steps_this_session is not None and (
        type(max_steps_this_session) is not int or max_steps_this_session < 1
    ):
        raise ValueError("Session step limit must be a positive integer")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new run directory, including for resumed sessions")
    data = SegmentationTask(task, manifest, source, allow_unverified=allow_unverified)
    dataset = data.dataset("train", config["size"])
    seed_everything(config["seed"])
    model = task_model(task, config).to(dev)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config["steps"])
    history = []
    start_step = 0
    parent_hash = None
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=True)
        if (
            state.get("format") != "ocuweave-task-v1"
            or state.get("task") != task
            or state.get("manifest_sha256") != data.manifest_sha256
            or state.get("config") != config
            or state.get("classes") != list(data.classes)
        ):
            raise ValueError(
                "Resume requires the same task, data, classes and exact configuration"
            )
        if state.get("software_version") != __version__:
            raise ValueError("Resume with the original software version")
        if (
            state.get("diagnostic")
            or state.get("complete")
            or not 0 < state.get("step", 0) < config["steps"]
        ):
            raise ValueError("Only interrupted runs can resume")
        if state.get("device_type") != dev.type:
            raise ValueError("Keep the original device type when resuming")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        torch.set_rng_state(state["rng_cpu"])
        if dev.type == "cuda":
            torch.cuda.set_rng_state_all(state["rng_cuda"])
        history = state["history"]
        start_step = state["step"]
        parent_hash = sha256(resume)
        if len(history) != start_step or history[-1]["step"] != start_step:
            raise ValueError("Checkpoint history is inconsistent")
    end = (
        min(config["steps"], start_step + max_steps_this_session)
        if max_steps_this_session
        else config["steps"]
    )
    output.mkdir(parents=True, exist_ok=False)
    provenance = {
        **software_provenance(),
        "task": task,
        "manifest_sha256": data.manifest_sha256,
        "config": config,
        "device": str(dev),
        "device_name": (
            torch.cuda.get_device_name(dev) if dev.type == "cuda" else dev.type
        ),
        "torch": str(torch.__version__),
        "precision": "float32",
        "classes": list(data.classes),
        "resume_sha256": parent_hash,
        "protocol": "fixed-step-task-v1",
        "checkpoint_selection": "fixed final step; no validation selection",
        "augmentation": "step-seeded paired horizontal flip; partial final batch retained",
        "test_evaluated": False,
        "start_step": start_step,
    }
    write_json(output / "provenance.json", provenance)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    started = time.monotonic()
    try:
        model.train()
        for index in range(start_step, end):
            batch = prepare_batch(
                dataset,
                step_indices(len(dataset), config["batch_size"], config["seed"], index),
                dev,
            )
            batch = horizontal_flip(
                batch,
                seed=config["seed"] + index,
                probability=config["flip_probability"],
            )
            optimizer.zero_grad(set_to_none=True)
            components = observed_components(
                model(batch["image"]), batch["mask"], batch["valid"], batch["available"]
            )
            components["total"].backward()
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config["gradient_clip"], error_if_nonfinite=True
            )
            if not torch.isfinite(components["total"]) or norm <= 0:
                raise FloatingPointError("Nonfinite loss or zero gradient")
            optimizer.step()
            scheduler.step()
            step = index + 1
            history.append(
                {
                    "step": step,
                    **{k: float(v.detach()) for k, v in components.items()},
                    "gradient_norm": float(norm),
                    "batch_size": len(batch["id"]),
                }
            )
            if step % config["save_every"] == 0 or step == end:
                state = {
                    "format": "ocuweave-task-v1",
                    "software_version": __version__,
                    "task": task,
                    "classes": list(data.classes),
                    "config": config,
                    "manifest_sha256": data.manifest_sha256,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "step": step,
                    "complete": step == config["steps"],
                    "history": history,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": (
                        torch.cuda.get_rng_state_all() if dev.type == "cuda" else None
                    ),
                    "device_type": dev.type,
                    "provenance": provenance,
                }
                atomic_checkpoint(output / "last.pt", state)
                write_json(output / "history.json", history)
        model.eval()
        with torch.no_grad():
            expected = model(batch["image"])
        reloaded = task_model(task, config).to(dev)
        reloaded.load_state_dict(
            torch.load(output / "last.pt", map_location="cpu", weights_only=True)[
                "model"
            ]
        )
        reloaded.eval()
        with torch.no_grad():
            torch.testing.assert_close(expected, reloaded(batch["image"]))
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        receipt = {
            "complete": end == config["steps"],
            "steps_completed": end,
            "session_steps": end - start_step,
            "checkpoint": str(output / "last.pt"),
            "checkpoint_sha256": sha256(output / "last.pt"),
            "checkpoint_reload_matches": True,
            "elapsed_seconds": time.monotonic() - started,
            "peak_cuda_allocated_bytes": (
                torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
            ),
            "peak_cuda_reserved_bytes": (
                torch.cuda.max_memory_reserved(dev) if dev.type == "cuda" else None
            ),
            "parameters": sum(p.numel() for p in model.parameters()),
            "test_evaluated": False,
            "validation_evaluated": False,
            "resume_sha256": parent_hash,
        }
        write_json(
            output / ("completed.json" if receipt["complete"] else "interrupted.json"),
            receipt,
        )
        return receipt
    except Exception as error:
        write_json(
            output / "failed.json",
            {
                "status": "failed",
                "error_type": type(error).__name__,
                "steps_completed": len(history),
                "last_checkpoint_exists": (output / "last.pt").exists(),
                "test_evaluated": False,
            },
        )
        raise
