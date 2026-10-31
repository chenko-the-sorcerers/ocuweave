"""Explicit versioned configuration for new resumable task runs."""

import math
from ocuweave.models.factory import make_model


def default_config():
    return dict(
        architecture="query_detail",
        width=24,
        dim=48,
        queries=8,
        size=384,
        batch_size=8,
        steps=500,
        seed=42,
        learning_rate=0.001,
        weight_decay=0.0001,
        gradient_clip=1.0,
        flip_probability=0.5,
        save_every=50,
    )


def validate_config(config):
    if set(config) != set(default_config()):
        raise ValueError("Config keys must match task-config exactly")
    if config["architecture"] not in {"query", "query_detail"}:
        raise ValueError("Choose query or query_detail")
    for key in ("width", "dim", "queries", "size", "batch_size", "steps", "save_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(key + " must be a positive integer")
    if (
        config["width"] < 4
        or config["dim"] % 4
        or config["size"] < 16
        or config["size"] % 16
    ):
        raise ValueError("width >= 4; dim divisible by 4; size >=16 divisible by 16")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("Nonnegative integer seed required")
    for key in ("learning_rate", "weight_decay", "gradient_clip", "flip_probability"):
        if (
            isinstance(config[key], bool)
            or not isinstance(config[key], (float, int))
            or not math.isfinite(config[key])
        ):
            raise ValueError("Finite numeric " + key + " required")
    if (
        not 0 < config["learning_rate"] < 1
        or config["weight_decay"] < 0
        or config["gradient_clip"] <= 0
        or not 0 <= config["flip_probability"] <= 1
    ):
        raise ValueError("Invalid optimizer or augmentation settings")


def task_model(task, config):
    if task == "idrid":
        from ocuweave.models.multilabel import MultiLabelLQNet

        return MultiLabelLQNet(
            classes=4,
            width=config["width"],
            dim=config["dim"],
            queries=config["queries"],
            local_detail=config["architecture"] == "query_detail",
        )
    if task not in {"fives", "otfid"}:
        raise ValueError("Unsupported segmentation task")
    return make_model(config)
