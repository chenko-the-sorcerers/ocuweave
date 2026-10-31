"""Synthetic fixtures exercise public task commands; no ophthalmic data or claims."""

from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ocuweave.utils.provenance import write_json


def create_task_fixture(task, output, seed=42):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(seed)
    if task not in {"fives", "idrid"}:
        raise ValueError("Fixture task must be fives or idrid")
    for i in range(14):
        training = i < 12
        image = rng.integers(10, 150, (32, 48, 3), dtype=np.uint8)
        if task == "fives":
            parent = output / ("train" if training else "test")
            name = f"{i}_N.png"
            mask = np.zeros((32, 48), dtype=np.uint8)
            mask[3:29, 15:18] = 255
            mask[12:15, 8:34] = 255
            image[mask > 0] = [220, 180, 100]
            for folder, array in [("Original", image), ("Ground truth", mask)]:
                (parent / folder).mkdir(parents=True, exist_ok=True)
                Image.fromarray(array).save(parent / folder / name)
        else:
            split = "a. Training Set" if training else "b. Testing Set"
            name = f"IDRiD_{i+1:02}"
            for c, label in enumerate(("MA", "HE", "EX", "SE")):
                if label == "SE" and i % 2 == 0:
                    continue
                mask = np.zeros((32, 48), dtype=np.uint8)
                mask[4 + c * 6 : 8 + c * 6, 8 + c * 8 : 12 + c * 8] = 255
                image[mask > 0] = [200 + c * 10, 170, 80]
                folder = (
                    output
                    / "A. Segmentation/2. All Segmentation Groundtruths"
                    / split
                    / label
                )
                folder.mkdir(parents=True, exist_ok=True)
                Image.fromarray(mask).save(folder / (name + "_" + label + ".tif"))
            folder = output / "A. Segmentation/1. Original Images" / split
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(image).save(folder / (name + ".jpg"))
    from ocuweave.data.adapters.fives import prepare_fives
    from ocuweave.data.adapters.idrid import prepare_idrid

    manifest = (prepare_fives if task == "fives" else prepare_idrid)(
        output, output / "manifest.json", seed=seed
    )
    manifest.update(
        synthetic=True,
        license="MIT",
        source="generated geometric fixture, not a clinical cohort",
    )
    write_json(output / "manifest.json", manifest)
    return output / "manifest.json"


def run_task_demo(output, device="cpu"):
    from ocuweave.training.task_config import default_config
    from ocuweave.training.tasks import fit
    from ocuweave.evaluation.tasks import evaluate_task
    from ocuweave.inference.tasks import infer
    from ocuweave.diagnostics.readiness import diagnostics

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        **default_config(),
        "width": 8,
        "dim": 16,
        "queries": 2,
        "size": 32,
        "steps": 2,
        "batch_size": 2,
        "save_every": 1,
    }
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    reports = {}
    try:
        for task in ("fives", "idrid"):
            folder = output / task
            manifest = create_task_fixture(task, folder / "data")
            report = fit(
                task,
                manifest,
                folder / "data",
                folder / "run",
                config,
                device=device,
                allow_unverified=True,
            )
            score = evaluate_task(
                task,
                manifest,
                folder / "data",
                report["checkpoint"],
                folder / "validation",
                device=device,
                allow_unverified=True,
            )
            data = __import__("json").loads(manifest.read_text())
            row = next(r for r in data["records"] if r["split"] == "validation")
            infer(
                report["checkpoint"],
                folder / "data" / row["image"],
                folder / "prediction",
                device=device,
            )
            check = diagnostics(
                task,
                manifest,
                folder / "data",
                folder / "diagnostics",
                config,
                device=device,
                allow_unverified=True,
            )
            reports[task] = {
                "completed": report["complete"],
                "validation_classes": list(score["per_class"]),
                "diagnostics_status": check["status"],
                "gpu_executed": check["gpu_executed"],
                "test_evaluated": False,
            }
    finally:
        torch.set_num_threads(previous)
    result = {
        "synthetic": True,
        "purpose": "software integration checks, not medical results",
        "tasks": reports,
    }
    write_json(output / "demo.json", result)
    return result


def run_multimodal_demo(output):
    from ocuweave.models.ocuweave import OcuWeave, ModelInputs
    from ocuweave.models.language.notes import tokenize_notes
    from ocuweave.losses.segmentation import segmentation_loss

    output = Path(output)
    if output.exists():
        raise FileExistsError("Choose a new synthetic demo directory")
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    try:
        model = OcuWeave(
            clinical_features=3, width=8, dim=16, queries=2, max_note_length=32
        )
        images = torch.rand(2, 3, 32, 32)
        target = (images[:, :1] > 0.7).float()
        ids, valid = tokenize_notes(["synthetic shape observation", None], 32)
        values = torch.tensor([[1.0, 2.0, float("nan")], [float("nan")] * 3])
        observed = torch.isfinite(values)
        model.clinical.fit_statistics(values, observed, splits=["train", "train"])
        result = model(
            ModelInputs(
                images,
                note_ids=ids,
                note_valid=valid,
                clinical_values=values,
                clinical_observed=observed,
            )
        )
        loss = segmentation_loss(result.predictions, target)
        loss.backward()
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite synthetic loss")
        output.mkdir(parents=True, exist_ok=False)
        report = {
            "synthetic": True,
            "purpose": "multimodal forward/backward contract check only",
            "loss": float(loss.detach()),
            "shape": list(result.predictions.shape),
            "evidence_available": result.text_evidence["available"].tolist(),
            "trained_on_clinical_records": False,
            "pretrained_llm": False,
            "clinical_validity_established": False,
        }
        write_json(output / "demo.json", report)
        return report
    finally:
        torch.set_num_threads(previous)
