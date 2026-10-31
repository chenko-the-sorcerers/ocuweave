"""Validated dataset access shared by CLI training, evaluation and diagnostics."""

import json
import re
import zipfile
from pathlib import Path
import numpy as np
from ocuweave.utils.provenance import sha256


def contained_path(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("Dataset paths must be relative to the data root")
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Dataset symlink escapes its root")
    return resolved


def verify_idrid(root, manifest):
    from ocuweave.data.adapters.idrid import CLASSES

    if manifest.get("protocol") != "idrid-four-channel-v1" or manifest.get(
        "classes"
    ) != list(CLASSES):
        raise ValueError("Expected an audited IDRiD four-channel manifest")
    groups, pixels = {}, {}
    for row in manifest["records"]:
        for key, mapping in (
            (row["similarity_group"], groups),
            (row["pixel_sha256"], pixels),
        ):
            if key in mapping and mapping[key] != row["split"]:
                raise ValueError("Cross-split group or duplicate")
            mapping[key] = row["split"]
        if sha256(contained_path(root, row["image"])) != row["image_sha256"]:
            raise ValueError("Image changed since audit")
        if set(row["masks"]) != set(CLASSES):
            raise ValueError("Missing class availability entries")
        availability = [row["masks"][c] is not None for c in CLASSES]
        if row["available"] != availability:
            raise ValueError("Annotation availability mismatch")
        for name, relative in row["masks"].items():
            if (
                relative is not None
                and sha256(contained_path(root, relative)) != row["mask_sha256"][name]
            ):
                raise ValueError("Annotation changed since audit")


class SegmentationTask:
    """Data is verified before any optimizer update; targets never become inputs."""

    def __init__(self, task, manifest, source, *, allow_unverified=False):
        from ocuweave.data.adapters.idrid import CLASSES

        self.task, self.manifest_path, self.source = task, Path(manifest), Path(source)
        self.manifest = json.loads(self.manifest_path.read_text())
        rows = self.manifest.get("records", [])
        ids = [r["sample_id"] for r in rows]
        if (
            not rows
            or len(ids) != len(set(ids))
            or any(
                not isinstance(s, str)
                or not re.fullmatch(r"[A-Za-z0-9_.-]+", s)
                or s in {".", ".."}
                for s in ids
            )
        ):
            raise ValueError("Unique, filename-safe sample identifiers required")
        if {r["split"] for r in rows} != {"train", "validation", "test"}:
            raise ValueError("Three nonempty audited splits required")
        if not self.manifest.get("patient_identity_verified") and not allow_unverified:
            raise ValueError(
                "Use --allow-unverified-patients only for exploratory image grouping"
            )
        if task == "otfid":
            from ocuweave.training.engine import checked_manifest

            self.manifest = checked_manifest(manifest, source, allow_unverified)
            self.classes = ("lesion",)
        elif task == "fives":
            from ocuweave.data.adapters.fives import verify_fives

            if self.manifest.get("protocol") != "fives-development-v1":
                raise ValueError("Expected FIVES manifest")
            for row in rows:
                for key in ("image", "mask"):
                    contained_path(source, row[key])
            verify_fives(source, self.manifest)
            self.classes = ("vessel",)
        elif task == "idrid":
            verify_idrid(source, self.manifest)
            self.classes = CLASSES
        else:
            raise ValueError("Supported tasks: otfid, fives, idrid")
        self.manifest_sha256 = sha256(manifest)

    def dataset(self, split, size):
        if split not in {"train", "validation"}:
            raise ValueError("Development workflow never loads test samples")
        if self.task == "otfid":
            from ocuweave.data.segmentation import SegmentationDataset

            return SegmentationDataset(
                self.manifest_path, self.source, split, size, False
            )
        if self.task == "fives":
            from ocuweave.data.adapters.fives import FivesDataset

            return FivesDataset(self.source, self.manifest, split, size, False)
        from ocuweave.data.adapters.idrid import IdridDataset

        return IdridDataset(self.source, self.manifest, split, size, False)

    def original(self, row):
        if row["split"] not in {"train", "validation"}:
            raise ValueError("Development-only data access")
        if self.task == "idrid":
            from ocuweave.data.adapters.idrid import read_idrid

            return read_idrid(self.source, row)
        if self.task == "fives":
            from ocuweave.data.adapters.fives import read_pair

            image, target = read_pair(self.source, row)
        else:
            from ocuweave.data.auditing.annotation_geometry import source_pair

            with zipfile.ZipFile(self.source) as archive:
                image, target, _ = source_pair(archive, row)
        return image, target[None], np.ones(1, dtype=bool)
