"""Registered native FIVES patch diagnostic, queued after the native matrix.

This is a sampling/context comparison, not a resolution-only causal estimate.
Uses standard library + Pillow and the frozen standalone C++ executable.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image
from native_data import export, load_case, sha256
from native_evaluate import overlap
from native_tiles import infer


def run(repo, build, session, root, manifest, wait=False):
    contract_path = session / "patch-contract.json"
    contract = json.loads(contract_path.read_text())
    device = contract.get("device", "cuda")
    if device not in ("cpu", "cuda"):
        raise ValueError("Unknown execution device")
    status_path = session / "patch-status.json"
    if status_path.exists():
        raise FileExistsError("Preserve existing patch programme status")
    status = {"status": "waiting_for_core", "contract_sha256": sha256(contract_path)}

    def persist():
        temp = status_path.with_suffix(".tmp")
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)

    persist()
    core_path = session / "programme-status.json"
    while True:
        core = json.loads(core_path.read_text())
        if core.get("status") == "completed":
            break
        if any(r["status"] in ("failed", "evaluation_failed") for r in core["runs"]):
            raise RuntimeError("Core programme failed; review before patch training")
        if not wait:
            raise RuntimeError("Core programme is not complete")
        # This is a user-launched background worker, not an agent tool wait.
        time.sleep(30)
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if revision != contract["source_commit"]:
        raise ValueError("Frozen native source differs")
    if subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip():
        raise ValueError("Frozen native checkout is dirty")
    if sha256(manifest) != contract["manifest_sha256"]:
        raise ValueError("Manifest differs")
    batch = session / "fives-patches.bin"
    meta = export(
        root,
        manifest,
        batch,
        contract["ids"],
        contract["input_size"],
        "fives",
        contract["patch_size"],
    )
    status.update(
        status="training", batch_sha256=meta["batch_sha256"], source_commit=revision
    )
    persist()
    output = session / "fives-patches"
    command = [
        str(build / "ocuweave-lqnet"),
        "--batch",
        str(batch),
        "--classes",
        "1",
        "--device",
        device,
        "--output",
        str(output),
        "--steps",
        str(contract["steps"]),
        "--save-every",
        "25",
        "--vessels",
        "true",
    ]
    started = time.time()
    with (session / "fives-patches.log").open("x") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    status["training_wall_seconds"] = time.time() - started
    baseline = session / "fives-baseline"
    if sha256(baseline / "initial.ckpt") != sha256(output / "initial.ckpt"):
        raise ValueError("Patch/base initial model and optimizer differ")
    status.update(status="reconstructing", initial_checkpoint_identical=True)
    persist()
    full_meta = json.loads((session / "fives.json").read_text())
    rows = []
    # Evaluate BOTH the full-image-trained and patch-trained checkpoints with
    # the same tiled inference context. Original full-image inference is already
    # reported by the core programme, allowing inference-context effects to be seen.
    for model_name, checkpoint in (
        ("full_image_trained", baseline / "latest.ckpt"),
        ("patch_trained", output / "latest.ckpt"),
    ):
        for sample in full_meta["samples"]:
            image, masks = load_case(root, sample["source"], ["vessel"])
            source = root / sample["source"]["image"]
            destination = session / ("fives-tiled-" + model_name) / sample["source_id"]
            destination.parent.mkdir(exist_ok=True)
            started = time.time()
            infer(
                build / "ocuweave-predict",
                checkpoint,
                source,
                destination,
                1,
                size=contract["input_size"],
                tile_size=contract["patch_size"],
                overlap=contract["inference_overlap"],
                device=device,
            )
            with Image.open(destination / "class-0-mask.png") as m:
                pred = m.convert("L")
            scores = overlap(pred, masks[0])
            rows.append(
                {
                    "model": model_name,
                    "sample_id": sample["source_id"],
                    "metrics": scores,
                    "image_area_pixels": image.width * image.height,
                    "predicted_image_area_percent": 100
                    * sum(pred.histogram()[1:])
                    / (image.width * image.height),
                    "false_positive_image_area_percent": 100
                    * scores["fp"]
                    / (image.width * image.height),
                    "inference_wall_seconds": time.time() - started,
                }
            )
            status["reconstructed_images"] = len(rows)
            persist()
    report = {
        "scope": "training-only patch diagnostic",
        "grid": "original image",
        "source_commit": revision,
        "contract_sha256": sha256(contract_path),
        "rows": rows,
        "area_denominator": "whole image, not verified retina/FOV",
        "early_detection_validated": False,
        "limitation": "Two selected patches per source vs one full image per source per step; sampling, context, and exposure change together. No resolution-only attribution.",
    }
    (session / "patch-results.json").write_text(json.dumps(report, indent=2) + "\n")
    status.update(status="completed", results="patch-results.json")
    persist()
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "build", "session", "root", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--wait", action="store_true")
    a = p.parse_args()
    try:
        run(a.repo, a.build, a.session, a.root, a.manifest, a.wait)
    except Exception as exc:
        failure = a.session / "patch-failure.json"
        if not failure.exists():
            failure.write_text(
                json.dumps({"type": type(exc).__name__, "message": str(exc)}, indent=2)
                + "\n"
            )
        raise
