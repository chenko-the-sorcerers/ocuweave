"""Summarize native diagnostic receipts without treating unfinished runs as scores.

Only reads saved numerical receipts; excludes raw images, masks and checkpoints.
All original-grid statistics are derived from the recorded confusion counts.
"""

import argparse
import csv
import datetime
import json
import math
import time
from pathlib import Path


def summarize(session, output):
    if output.exists():
        raise FileExistsError("Choose a new immutable report directory")
    contract = json.loads((session / "contract-locked.json").read_text())
    state = json.loads((session / "programme-status.json").read_text())
    if state["source_commit"] != contract["source_commit"]:
        raise ValueError("Contract/status source mismatch")
    registered = [
        (task, run["name"])
        for task in ("idrid", "fives")
        for run in contract["runs"]
        if run.get("task", task) == task
    ]
    observed = {(r["task"], r["name"]): r for r in state["runs"]}
    rows, runs = [], []
    for task, name in registered:
        item = observed.get((task, name), {"status": "queued"})
        run = {"task": task, "name": name, "status": item["status"]}
        runs.append(run)
        directory = session / (task + "-" + name)
        if item["status"] != "completed":
            history = directory / "history.jsonl"
            if history.exists():
                lines = history.read_text().splitlines()
                if lines:
                    try:
                        run["last_observed_step"] = json.loads(lines[-1])["step"]
                    except json.JSONDecodeError:
                        pass  # A concurrently written last line is not a result.
            continue
        receipt = json.loads((directory / "status.json").read_text())
        protocol = json.loads((directory / "protocol.json").read_text())
        if (
            not receipt["checkpoint_reload_matches"]
            or protocol["source_commit"] != contract["source_commit"]
            or protocol["source_dirty"]
        ):
            raise ValueError("Invalid completion/source receipt")
        metrics = json.loads((directory / "original-grid/metrics.json").read_text())
        input_metrics = json.loads((directory / "metrics.json").read_text())
        memory = json.loads(
            (session / (task + "-" + name + "-memory.json")).read_text()
        )
        samples = memory["samples"]
        run.update(
            parameters=protocol["parameters"],
            elapsed_seconds=receipt["elapsed_seconds"],
            steps=protocol["start_step"] + protocol["additional_steps"],
            checkpoint_sha256=item["checkpoint_sha256"],
            batch_sha256=item["batch_sha256"],
            sampled_gpu_mib=max(
                (
                    s["gpu_process_mib"]
                    for s in samples
                    if s["gpu_process_mib"] is not None
                ),
                default=None,
            ),
            sampled_host_rss_kib=max(
                (s["host_rss_kib"] for s in samples if s["host_rss_kib"] is not None),
                default=None,
            ),
        )
        for j, (label, summary) in enumerate(metrics["summary"].items()):
            group = [
                r
                for r in metrics["rows"]
                if r["class"] == label and r["metrics"] is not None
            ]
            counts = {
                key: sum(r["metrics"][key] for r in group) for key in ("tp", "fp", "fn")
            }
            for r in group:
                m = r["metrics"]
                denom = 2 * m["tp"] + m["fp"] + m["fn"]
                dice = 2 * m["tp"] / denom if denom else 1.0
                if not math.isclose(m["dice"], dice, abs_tol=1e-12):
                    raise ValueError("Recorded Dice disagrees with confusion counts")
                denom = m["tp"] + m["fp"] + m["fn"]
                iou = m["tp"] / denom if denom else 1.0
                if not math.isclose(m["iou"], iou, abs_tol=1e-12):
                    raise ValueError("Recorded IoU disagrees with confusion counts")
            tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
            input_group = [
                r
                for r in input_metrics["rows"]
                if r["class_index"] == j and r["annotation_available"]
            ]
            cl = [r["cldice"] for r in input_group if "cldice" in r]
            rows.append(
                {
                    "task": task,
                    "run": name,
                    "class": label,
                    "steps": run["steps"],
                    **summary,
                    "pooled_precision": tp / (tp + fp) if tp + fp else None,
                    "pooled_recall": tp / (tp + fn) if tp + fn else None,
                    "false_positive_pixels": fp,
                    "false_positive_image_area_percent": (
                        100 * fp / sum(r["image_area_pixels"] for r in group)
                        if group
                        else None
                    ),
                    "input_grid_cldice_macro": sum(cl) / len(cl) if cl else None,
                    "elapsed_seconds": run["elapsed_seconds"],
                    "sampled_gpu_mib": run["sampled_gpu_mib"],
                }
            )
    report = {
        "captured_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_commit": state["source_commit"],
        "scope": "training-only diagnostic; not validation/generalization",
        "core_complete": all(r["status"] == "completed" for r in runs),
        "runs": runs,
        "class_results": rows,
        "memory_scope": "sampled process memory; not allocator peak",
        "early_detection_validated": False,
    }
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    if rows:
        with (output / "metrics.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def watch(session, output):
    """Save a new numerical report at each run-state transition; stop at completion/failure."""
    output.mkdir(parents=True, exist_ok=False)
    previous = None
    while True:
        state = json.loads((session / "programme-status.json").read_text())
        key = [(r["task"], r["name"], r["status"]) for r in state["runs"]]
        if key != previous:
            stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y%m%dT%H%M%S%fZ"
            )
            report = summarize(session, output / stamp)
            print(
                json.dumps({"report": stamp, "core_complete": report["core_complete"]}),
                flush=True,
            )
            previous = key
            if report["core_complete"] or any(
                r["status"] in ("failed", "evaluation_failed") for r in state["runs"]
            ):
                return report
        time.sleep(30)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--session", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--watch",
        action="store_true",
        help="Write a new report after each run-state transition",
    )
    a = p.parse_args()
    print(json.dumps((watch if a.watch else summarize)(a.session, a.output), indent=2))
