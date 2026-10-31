"""Installed command-line interface; imports research dependencies only on use."""

import argparse
import json
from pathlib import Path

from ocuweave import __version__


def parser():
    root = argparse.ArgumentParser(
        prog="ocuweave", description="OcuWeave ophthalmic research toolkit"
    )
    root.add_argument("--version", action="version", version=f"OcuWeave {__version__}")
    commands = root.add_subparsers(dest="command", required=True)

    config = commands.add_parser(
        "init-config", help="Write an explicit training configuration"
    )
    config.add_argument("--output", required=True)
    config.add_argument(
        "--architecture",
        choices=["query", "query_detail", "fcn"],
        default="query_detail",
    )
    config.add_argument("--size", type=int, choices=[192, 384], default=192)
    config.add_argument("--seed", type=int, default=42)

    demo = commands.add_parser(
        "demo",
        help="Train, evaluate and predict on generated geometric shapes (not retinal data)",
    )
    demo.add_argument(
        "--output",
        required=True,
        help="New directory; existing artifacts are never overwritten",
    )
    demo.add_argument(
        "--architecture",
        choices=["query", "query_detail", "fcn"],
        default="query_detail",
    )
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--epochs", type=int, default=2)
    demo.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"])

    prepare = commands.add_parser(
        "prepare-otfid",
        help="Audit an authorized OTFID ZIP and create a frozen manifest",
    )
    prepare.add_argument("--archive", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--patient-map")
    prepare.add_argument("--image-column")
    prepare.add_argument("--label-column")
    prepare.add_argument("--allow-unverified-patients", action="store_true")

    for name in ("train", "evaluate"):
        command = commands.add_parser(
            name,
            help=f"{name.capitalize()} using an audited manifest and local archive",
        )
        for argument in ("manifest", "archive", "output"):
            command.add_argument("--" + argument, required=True)
        command.add_argument(
            "--device", default="auto", choices=["cpu", "cuda", "mps", "auto"]
        )
        command.add_argument("--allow-unverified-patients", action="store_true")
        if name == "train":
            command.add_argument("--config", required=True)
        else:
            command.add_argument("--checkpoint", required=True)
            command.add_argument(
                "--split", choices=["validation", "test"], default="validation"
            )
            command.add_argument(
                "--grid",
                choices=["original", "input"],
                default="original",
                help="Original-grid evaluation is development-only; legacy test uses input grid",
            )

    prediction = commands.add_parser(
        "predict", help="Save original-size probability, binary mask and provenance"
    )
    for argument in ("checkpoint", "image", "output"):
        prediction.add_argument("--" + argument, required=True)
    prediction.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda", "mps", "auto"]
    )
    task_config = commands.add_parser(
        "task-config", help="Write a locked fixed-step task configuration"
    )
    task_config.add_argument("--output", required=True)
    task_config.add_argument("--size", type=int, default=384)
    task_config.add_argument("--steps", type=int, default=500)
    task_config.add_argument("--seed", type=int, default=42)
    task_config.add_argument(
        "--architecture", choices=["query", "query_detail"], default="query_detail"
    )
    for task in ("fives", "idrid"):
        command = commands.add_parser(
            "prepare-" + task,
            help="Audit the extracted " + task.upper() + " segmentation release",
        )
        command.add_argument("--root", required=True)
        command.add_argument("--manifest", required=True)
        command.add_argument("--seed", type=int, default=42)
        if task == "fives":
            command.add_argument("--provisional-development", action="store_true")
    for name in ("fit", "score", "diagnose"):
        command = commands.add_parser(
            name,
            help={
                "fit": "New resumable fixed-step task training",
                "score": "Original-grid validation bound to a checkpoint manifest",
                "diagnose": "Training-only small-set fitting and gradient/probability diagnostics",
            }[name],
        )
        command.add_argument(
            "--task", required=True, choices=["otfid", "fives", "idrid"]
        )
        for key in ("manifest", "source", "output"):
            command.add_argument("--" + key, required=True)
        command.add_argument(
            "--device",
            choices=["cpu", "cuda", "mps", "auto"],
            default="cpu" if name == "score" else "cuda",
        )
        command.add_argument("--allow-unverified-patients", action="store_true")
        if name == "score":
            command.add_argument("--checkpoint", required=True)
        else:
            command.add_argument("--config", required=True)
        if name == "fit":
            command.add_argument("--resume")
            command.add_argument("--max-steps-this-session", type=int)
        if name == "diagnose":
            command.add_argument("--count", type=int, choices=[2, 3, 4], default=2)
            command.add_argument("--sample-ids", nargs="+")
    command = commands.add_parser(
        "infer",
        help="Task-aware checkpoint inference with optional full-coverage tiling",
    )
    for key in ("checkpoint", "image", "output"):
        command.add_argument("--" + key, required=True)
    command.add_argument(
        "--device", choices=["cpu", "cuda", "mps", "auto"], default="cpu"
    )
    command.add_argument("--tile-size", type=int)
    command.add_argument("--overlap", type=int, default=0)
    for name in ("demo-tasks", "demo-multimodal"):
        command = commands.add_parser(
            name, help="Synthetic integration checks; no medical data"
        )
        command.add_argument("--output", required=True)
        if name == "demo-tasks":
            command.add_argument(
                "--device", choices=["cpu", "cuda", "mps"], default="cpu"
            )
    commands.add_parser(
        "doctor", help="Inspect environment and available devices without training"
    )
    return root


def dispatch(args):
    if args.command == "doctor":
        import importlib.metadata
        import platform

        result = {
            "python": platform.python_version(),
            "software_version": __version__,
            "packages": {},
        }
        for name in ("torch", "numpy", "Pillow", "scipy", "scikit-image"):
            try:
                result["packages"][name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                result["packages"][name] = None
        if result["packages"]["torch"]:
            import torch

            result.update(
                cuda_available=torch.cuda.is_available(),
                mps_available=torch.backends.mps.is_available(),
                gpu_name=(
                    torch.cuda.get_device_name() if torch.cuda.is_available() else None
                ),
            )
        return result
    if args.command == "task-config":
        from ocuweave.training.task_config import default_config, validate_config

        config = {
            **default_config(),
            "size": args.size,
            "steps": args.steps,
            "seed": args.seed,
            "architecture": args.architecture,
        }
        validate_config(config)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        return {"config": str(output), "protocol": "fixed-step-task-v1"}
    if args.command in {"prepare-fives", "prepare-idrid"}:
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        if args.command == "prepare-fives":
            from ocuweave.data.adapters.fives import prepare_fives

            data = prepare_fives(
                args.root,
                args.manifest,
                seed=args.seed,
                provisional_development=args.provisional_development,
            )
        else:
            from ocuweave.data.adapters.idrid import prepare_idrid

            data = prepare_idrid(args.root, args.manifest, seed=args.seed)
        return {"counts": data["counts"], "manifest": args.manifest}
    if args.command in {"fit", "score", "diagnose"}:
        options = dict(
            device=args.device, allow_unverified=args.allow_unverified_patients
        )
        if args.command == "score":
            from ocuweave.evaluation.tasks import evaluate_task

            report = evaluate_task(
                args.task,
                args.manifest,
                args.source,
                args.checkpoint,
                args.output,
                **options,
            )
            return {
                "task": args.task,
                "metrics": report["per_class"],
                "report": str(Path(args.output) / "metrics.json"),
            }
        config = json.loads(Path(args.config).read_text())
        if args.command == "fit":
            from ocuweave.training.tasks import fit

            return fit(
                args.task,
                args.manifest,
                args.source,
                args.output,
                config,
                resume=args.resume,
                max_steps_this_session=args.max_steps_this_session,
                **options,
            )
        from ocuweave.diagnostics.readiness import diagnostics

        report = diagnostics(
            args.task,
            args.manifest,
            args.source,
            args.output,
            config,
            count=args.count,
            sample_ids=args.sample_ids,
            **options,
        )
        return {
            "status": report["status"],
            "gpu_executed": report["gpu_executed"],
            "untested_classes": report["untested_classes"],
            "report": str(Path(args.output) / "diagnostics.json"),
        }
    if args.command == "infer":
        from ocuweave.inference.tasks import infer

        return infer(
            args.checkpoint,
            args.image,
            args.output,
            device=args.device,
            tile_size=args.tile_size,
            overlap=args.overlap,
        )
    if args.command == "demo-tasks":
        from ocuweave.demo_tasks import run_task_demo

        return run_task_demo(args.output, args.device)
    if args.command == "demo-multimodal":
        from ocuweave.demo_tasks import run_multimodal_demo

        return run_multimodal_demo(args.output)
    if args.command == "init-config":
        if args.seed < 0:
            raise ValueError("Seed must be nonnegative")
        config = dict(
            architecture=args.architecture,
            width=24,
            dim=48,
            queries=8,
            size=args.size,
            batch_size=8,
            epochs=20,
            learning_rate=0.001,
            seed=args.seed,
        )
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x") as stream:
            json.dump(config, stream, indent=2)
            stream.write("\n")
        return {"config": str(output)}
    if args.command == "demo":
        from ocuweave.demo import run_demo

        return run_demo(
            args.output,
            architecture=args.architecture,
            seed=args.seed,
            epochs=args.epochs,
            device=args.device,
        )
    if args.command == "prepare-otfid":
        from ocuweave.data.adapters.otfid import audit_otfid

        result = audit_otfid(
            args.archive,
            args.output,
            seed=args.seed,
            allow_unverified_patients=args.allow_unverified_patients,
            patient_map=args.patient_map,
            image_column=args.image_column,
            label_column=args.label_column,
        )
        return {
            key: result[key] for key in ("eligible_samples", "splits", "split_unit")
        }
    if args.command == "train":
        from ocuweave.training.engine import train

        checkpoint = train(
            args.manifest,
            args.archive,
            args.output,
            json.loads(Path(args.config).read_text()),
            allow_unverified=args.allow_unverified_patients,
            device=args.device,
        )
        return {"checkpoint": str(checkpoint)}
    if args.command == "evaluate":
        if args.grid == "original":
            if args.split != "validation":
                raise ValueError(
                    "Original-grid CLI evaluation is validation-only. Keep test data out of development."
                )
            from ocuweave.evaluation.original_grid import evaluate_original as evaluate
        else:
            from ocuweave.training.engine import evaluate
        result = evaluate(
            args.manifest,
            args.archive,
            args.checkpoint,
            args.output,
            split=args.split,
            allow_unverified=args.allow_unverified_patients,
            device=args.device,
        )
        return {
            "evaluation_grid": args.grid,
            "split": args.split,
            "metrics": result["metrics"],
        }
    if args.command == "predict":
        from ocuweave.inference.segmentation import save_prediction

        return save_prediction(
            args.checkpoint, args.image, args.output, device=args.device
        )
    raise ValueError("Unsupported command")


def main(argv=None):
    command_parser = parser()
    args = command_parser.parse_args(argv)
    try:
        result = dispatch(args)
    except ModuleNotFoundError as error:
        if error.name not in {
            "torch",
            "numpy",
            "PIL",
            "matplotlib",
            "scipy",
            "skimage",
        }:
            raise
        command_parser.error(
            'Research dependencies are missing. Install with: python -m pip install ".[research]" (from the repository).'
        )
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        command_parser.error(str(error))
    print(json.dumps(result, indent=2))
    return 0
