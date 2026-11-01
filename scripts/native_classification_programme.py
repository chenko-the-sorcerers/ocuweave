"""Run the registered native FIVES classification baseline on an available, validated GPU.

Only standard library/Pillow preparation and standalone C++/CUDA training.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path
from native_data import sha256
from native_classification_data import export
from native_classification_evaluate import evaluate


def execute(repo, build, session, root, manifest, expected_commit, wait_session=None):
    if session.exists():
        raise FileExistsError("Use a new immutable classification session")
    contract = json.loads(
        (repo / "configs/native/classification-fives.json").read_text()
    )
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if (
        actual != expected_commit
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
    ):
        raise ValueError("Frozen classifier source differs")
    if sha256(manifest) != contract["manifest_sha256"]:
        raise ValueError("Frozen FIVES manifest differs")
    session.mkdir(parents=True)
    status = {
        "status": "waiting_for_segmentation" if wait_session else "preparing",
        "source_commit": actual,
    }

    def persist():
        temporary = session / "status.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(session / "status.json")

    persist()
    contract["source_commit"] = actual
    contract["runner_sha256"] = sha256(Path(__file__))
    (session / "contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    try:
        if wait_session:
            while True:
                core = json.loads((wait_session / "programme-status.json").read_text())
                patch = json.loads((wait_session / "patch-status.json").read_text())
                if (
                    any(
                        r["status"] in ("failed", "evaluation_failed")
                        for r in core["runs"]
                    )
                    or (wait_session / "patch-failure.json").exists()
                ):
                    raise RuntimeError(
                        "Preceding programme failed; inspect its evidence before allocating new training"
                    )
                if core.get("status") == "completed" and patch["status"] == "completed":
                    break
                time.sleep(30)
        status["status"] = "validating_native_backend"
        persist()
        with (session / "native-validation.log").open("x") as log:
            subprocess.run(
                ["ctest", "--test-dir", str(build), "--output-on-failure"],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        cache = (build / "CMakeCache.txt").read_text()
        if "OCUWEAVE_ENABLE_CUDA:BOOL=ON" not in cache:
            raise ValueError("Registered medical run requires the validated CUDA build")
        status["status"] = "preparing_data"
        persist()
        data = session / "data"
        meta = export(root, manifest, data, contract["input_size"])
        for split in ("train", "validation"):
            for extension, field in [("bin", "batch_sha256"), ("tsv", "labels_sha256")]:
                if (
                    sha256(data / (split + "." + extension))
                    != meta["partitions"][split][field]
                ):
                    raise ValueError("Classifier inputs changed")
        output = session / "run"
        binary = build / "ocuweave-classify"
        command = [
            str(binary),
            "--train",
            str(data / "train.bin"),
            "--train-labels",
            str(data / "train.tsv"),
            "--validation",
            str(data / "validation.bin"),
            "--validation-labels",
            str(data / "validation.tsv"),
            "--output",
            str(output),
            "--steps",
            str(contract["steps"]),
            "--batch-size",
            str(contract["batch_size"]),
            "--width",
            str(contract["width"]),
            "--classes",
            str(len(contract["classes"])),
            "--seed",
            str(contract["seed"]),
            "--lr",
            str(contract["learning_rate"]),
            "--device",
            "cuda",
        ]
        status.update(
            status="training", command=command, executable_sha256=sha256(binary)
        )
        persist()
        with (session / "training.log").open("x") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        receipt = json.loads((output / "protocol.json").read_text())
        if receipt["source_commit"] != actual or receipt["source_dirty"]:
            raise ValueError(
                "Training binary was not built from the frozen clean source"
            )
        status["status"] = "recounting_validation"
        persist()
        report = evaluate(data, output, session / "evaluation")
        status.update(
            status="completed_requires_review",
            validation=report["results"]["validation"],
            validation_accuracy_target_met=report["validation_accuracy_target_met"],
            checkpoint_sha256=sha256(output / "latest.ckpt"),
        )
        persist()
        print(json.dumps(status, indent=2), flush=True)
    except Exception as exc:
        status.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        persist()
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "build", "session", "root", "manifest"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--expected-commit", required=True)
    p.add_argument("--wait-session", type=Path)
    a = p.parse_args()
    execute(
        a.repo,
        a.build,
        a.session,
        a.root,
        a.manifest,
        a.expected_commit,
        a.wait_session,
    )
