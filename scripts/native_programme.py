"""Execute the registered training-only native diagnostic matrix sequentially.

Uses only the Python standard library; the standalone executable owns training.
The source checkout/build must be frozen and CUDA tests passed by the caller.
"""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def execute(repo, build, session, roots, tasks):
    contract = json.loads((session / "contract-locked.json").read_text())
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if (
        actual != contract["source_commit"]
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
    ):
        raise ValueError("Source differs from frozen contract")
    executable = build / "ocuweave-lqnet"
    jobs = []
    for task in tasks:
        for run in contract["runs"]:
            if run.get("task", task) != task:
                continue
            jobs.append((task, run))
    status = {
        "source_commit": actual,
        "executable_sha256": digest(executable),
        "scope": "training-only",
        "runs": [],
    }
    status_path = session / "programme-status.json"

    def persist():
        temp = session / "programme-status.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(status_path)

    persist()
    for task, run in jobs:
        name = run["name"]
        output = session / (task + "-" + name)
        if output.exists():
            raise FileExistsError("Preserve prior run: " + str(output))
        batch = session / (task + ".bin")
        meta = json.loads(batch.with_suffix(".json").read_text())
        if digest(batch) != meta["batch_sha256"]:
            raise ValueError("Batch checksum changed")
        item = {
            "task": task,
            "name": name,
            "status": "running",
            "batch_sha256": digest(batch),
            "start_unix": time.time(),
        }
        status["runs"].append(item)
        persist()
        command = [
            str(executable),
            "--batch",
            str(batch),
            "--classes",
            str(len(meta["classes"])),
            "--device",
            "cuda",
            "--output",
            str(output),
            "--steps",
            str(run.get("steps", run.get("additional_steps", 200))),
            "--save-every",
            "25",
            "--variant",
            run.get("variant", "baseline"),
        ]
        if name == "weighted":
            command += [
                "--weights",
                ",".join(map(str, contract["idrid_positive_weights"])),
            ]
        if name == "longer":
            command += ["--resume", str(session / (task + "-baseline") / "latest.ckpt")]
        if task == "fives":
            command += ["--vessels", "true"]
        item["command"] = command
        persist()
        with (session / (task + "-" + name + ".log")).open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            samples = []
            while child.poll() is None:
                try:
                    gpu = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=pid,used_gpu_memory",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=5,
                    )
                    used = [
                        int(line.split(",")[1].strip())
                        for line in gpu.splitlines()
                        if int(line.split(",")[0].strip()) == child.pid
                    ]
                    proc = Path("/proc") / str(child.pid) / "status"
                    rss = (
                        next(
                            (
                                int(line.split()[1])
                                for line in proc.read_text().splitlines()
                                if line.startswith("VmRSS:")
                            ),
                            None,
                        )
                        if proc.exists()
                        else None
                    )
                    samples.append(
                        {
                            "elapsed": time.time() - item["start_unix"],
                            "gpu_process_mib": used[0] if used else None,
                            "host_rss_kib": rss,
                        }
                    )
                except (subprocess.SubprocessError, ValueError, OSError):
                    pass
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            item["returncode"] = child.returncode
        item["end_unix"] = time.time()
        item["status"] = "training_completed" if child.returncode == 0 else "failed"
        (session / (task + "-" + name + "-memory.json")).write_text(
            json.dumps(
                {
                    "sampling_interval_seconds": 5,
                    "scope": "process memory sampled, not allocator peak",
                    "samples": samples,
                },
                indent=2,
            )
        )
        persist()
        if child.returncode:
            raise RuntimeError("Native diagnostic failed: " + task + "/" + name)
        if name == "weighted":
            baseline_initial = session / (task + "-baseline") / "initial.ckpt"
            if digest(baseline_initial) != digest(output / "initial.ckpt"):
                raise ValueError("Paired objective initial states differ")
            item["paired_initial_checkpoint_identical"] = True
        item["initial_sha256"] = digest(output / "initial.ckpt")
        item["checkpoint_sha256"] = digest(output / "latest.ckpt")
        evaluation = [
            sys.executable,
            str(repo / "scripts/native_evaluate.py"),
            "--predictions",
            str(output / "predictions.bin"),
            "--metadata",
            str(batch.with_suffix(".json")),
            "--root",
            str(roots[task]),
            "--output",
            str(output / "original-grid"),
        ]
        with (output / "evaluation.log").open("x") as log:
            result = subprocess.run(evaluation, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            item["status"] = "evaluation_failed"
            persist()
            result.check_returncode()
        item["status"] = "completed"
        item["summary"] = json.loads(
            (output / "original-grid/metrics.json").read_text()
        )["summary"]
        persist()
        print(task, name, item["summary"], flush=True)
    status["status"] = "completed"
    persist()
    return status


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--build", type=Path, required=True)
    parser.add_argument("--session", type=Path, required=True)
    parser.add_argument("--idrid-root", type=Path, required=True)
    parser.add_argument("--fives-root", type=Path, required=True)
    parser.add_argument(
        "--tasks", nargs="+", choices=["idrid", "fives"], default=["idrid", "fives"]
    )
    a = parser.parse_args()
    execute(
        a.repo,
        a.build,
        a.session,
        {"idrid": a.idrid_root, "fives": a.fives_root},
        a.tasks,
    )
