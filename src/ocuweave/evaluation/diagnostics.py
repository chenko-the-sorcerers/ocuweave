"""Development-only failure analysis of frozen segmentation checkpoints."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ocuweave.data.segmentation import SegmentationDataset
from ocuweave.evaluation.segmentation import aggregate, binary_metrics
from ocuweave.models.factory import make_model
from ocuweave.training.engine import checked_manifest, select_device
from ocuweave.utils.provenance import sha256, write_json


def require_development_split(split):
    if split not in {"train", "validation"}:
        raise ValueError("Failure diagnostics accept training/validation only; test remains sealed")


def area_bin(fraction):
    if fraction == 0:
        return "empty"
    if fraction <= .01:
        return "small_le_1pct"
    if fraction <= .05:
        return "medium_1_to_5pct"
    return "large_gt_5pct"


def stratify(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    return {label: aggregate(items) for label, items in sorted(groups.items())}


@torch.no_grad()
def diagnose(manifest, archive, checkpoint, output, *, split="validation",
             allow_unverified=False, device="cpu"):
    require_development_split(split)
    output = Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior diagnostics; choose a new output")
    data = checked_manifest(manifest, archive, allow_unverified)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["provenance"]["manifest_sha256"] != sha256(manifest):
        raise ValueError("Checkpoint and manifest differ")
    dev = select_device(device)
    model = make_model(state["config"]).to(dev)
    model.load_state_dict(state["model"]); model.eval()
    dataset = SegmentationDataset(manifest, archive, split, state["config"]["size"])
    metadata = {r["sample_id"]: r for r in dataset.rows}
    rows = []
    for batch in DataLoader(dataset, batch_size=state["config"]["batch_size"]):
        probabilities = model(batch["image"].to(dev)).sigmoid().cpu().numpy()
        for i, sample_id in enumerate(batch["id"]):
            valid = batch["valid"][i].numpy().astype(bool)
            target = batch["mask"][i].numpy().astype(bool)
            pred = probabilities[i] >= .5
            info = metadata[sample_id]
            original_area = info["lesion_pixels"] / (info["height"] * info["width"])
            metrics = binary_metrics(pred, target, valid)
            rows.append({"sample_id": sample_id, "source": info.get("source") or "unknown",
                         "label": info["label"], "area_bin": area_bin(original_area),
                         "original_area_fraction": original_area,
                         "resized_lesion_pixels": int((target & valid).sum()),
                         "vanished_after_resize": bool(info["lesion_pixels"] > 0 and not (target & valid).any()),
                         "valid_pixels": int(valid.sum()),
                         "false_positive_area_fraction": metrics["fp"] / int(valid.sum()),
                         "precision": metrics["tp"] / (metrics["tp"] + metrics["fp"]) if metrics["tp"] + metrics["fp"] else None,
                         "recall": metrics["tp"] / (metrics["tp"] + metrics["fn"]) if metrics["tp"] + metrics["fn"] else None,
                         **metrics})
    positive = sorted((r for r in rows if r["target_positive"]), key=lambda r: (r["dice"], r["sample_id"]))
    healthy = sorted((r for r in rows if not r["target_positive"]), key=lambda r: (-r["false_positive_area_fraction"], r["sample_id"]))
    result = {"protocol": "otfid-development-errors-v1", "split": split,
              "checkpoint_sha256": sha256(checkpoint), "manifest_sha256": sha256(manifest),
              "archive_sha256": data["archive_sha256"], "checkpoint_epoch": state["epoch"],
              "training_source_commit": state["provenance"]["source_commit"],
              "size": state["config"]["size"], "threshold": .5,
              "metrics": aggregate(rows),
              "by_source": stratify(rows, "source"), "by_label": stratify(rows, "label"),
              "by_original_area": stratify(rows, "area_bin"),
              "vanished_after_resize": sum(r["vanished_after_resize"] for r in rows),
              "review_selection": {"lowest_positive_dice": [r["sample_id"] for r in positive[:3]],
                                   "largest_healthy_false_positive_area": [r["sample_id"] for r in healthy[:3]]},
              "per_image": rows,
              "limitations": ["Descriptive development-set diagnostics; no test-based tuning.",
                              "Validation already selected the checkpoint; these are not independent performance estimates.",
                              "Area fractions use the full original image, not a clinically annotated retinal field.",
                              "Small source/label strata and unverified patient identities limit interpretation."]}
    output.mkdir(parents=True)
    write_json(output / "diagnostics.json", result)
    plot_diagnostics(result, output / "strata.png")
    plot_failures(model.cpu(), dataset, result, output / "review_panel.png")
    return {key: value for key, value in result.items() if key != "per_image"}


def plot_diagnostics(result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = result["by_original_area"]
    labels = [key for key in groups if key != "empty"]
    values = [groups[key]["positive_macro_dice"] for key in labels]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].bar(range(len(labels)), [v or 0 for v in values], color="#2457A7")
    axes[0].set_xticks(range(len(labels)), [f"{label}\nn={groups[label]['positive_samples']}" for label in labels], fontsize=8)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel("Positive-image Dice")
    axes[0].set_title("By original lesion area / full image")
    empty = [r["false_positive_area_fraction"] * 100 for r in result["per_image"] if not r["target_positive"]]
    axes[1].hist(empty, bins=[0, .01, .1, 1, 5, 10, 25, 50, 100], color="#B56B14")
    axes[1].set_xlabel("False-positive area (% of valid image)")
    axes[1].set_ylabel("Healthy images"); axes[1].set_title(f"Healthy mask errors (n={len(empty)})")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{result['split']} diagnostics · fixed threshold 0.5 · no test tuning")
    fig.savefig(path, dpi=150); plt.close(fig)


@torch.no_grad()
def plot_failures(model, dataset, result, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    selected = list(dict.fromkeys(sum(result["review_selection"].values(), [])))
    index = {r["sample_id"]: i for i, r in enumerate(dataset.rows)}
    fig, axes = plt.subplots(len(selected), 3, figsize=(9, len(selected) * 2.5), squeeze=False, layout="constrained")
    for line, sample_id in enumerate(selected):
        sample = dataset[index[sample_id]]
        pred = model(sample["image"].unsqueeze(0)).sigmoid()[0, 0].numpy() >= .5
        pred &= sample["valid"][0].numpy().astype(bool)
        axes[line, 0].imshow(sample["image"].permute(1, 2, 0))
        axes[line, 0].set_title(sample_id)
        axes[line, 1].imshow(sample["mask"][0], cmap="gray", vmin=0, vmax=1)
        axes[line, 1].set_title("Reference mask")
        axes[line, 2].imshow(pred, cmap="gray", vmin=0, vmax=1)
        axes[line, 2].set_title("Prediction ≥ 0.5")
        for axis in axes[line]:
            axis.axis("off")
    fig.suptitle(f"{result['split']} failure review · deliberately selected errors, not representative samples")
    fig.savefig(path, dpi=130); plt.close(fig)
