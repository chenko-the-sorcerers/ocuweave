"""Export FIVES disease labels and images for native classification.

No vessel mask is read. The disease code is an annotation, never a model input.
Preserve the existing provisional train/validation groups; prohibit test export.
"""

import argparse
import json
import struct
from collections import Counter
from pathlib import Path
from PIL import Image, __version__ as pillow_version
from native_data import prepare, sha256, verified, write_float

CLASSES = {"A": "AMD", "D": "DR", "G": "glaucoma", "N": "normal"}
SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC9352679/"


def audit(manifest):
    rows = [r for r in manifest["records"] if r["split"] in ("train", "validation")]
    seen, groups, pixels = set(), {}, {}
    for row in rows:
        if row.get("official_split") != "train":
            raise ValueError("Development must come from official training partition")
        sid = row["sample_id"]
        if not sid or any(c in sid for c in ',\t\r\n"'):
            raise ValueError("Unsafe sample ID for TSV/CSV export")
        if sid in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(sid)
        code = Path(row["image"]).stem.rsplit("_", 1)[-1]
        if code not in CLASSES or code != row["disease_stratum"]:
            raise ValueError("Unknown/inconsistent official FIVES disease code")
        for key, mapping in [("similarity_group", groups), ("pixel_sha256", pixels)]:
            value = row[key]
            if not value:
                raise ValueError("Missing leakage audit evidence")
            if value in mapping and mapping[value] != row["split"]:
                raise ValueError("Cross-development group or duplicate")
            mapping[value] = row["split"]
    for split in ("train", "validation"):
        if {r["disease_stratum"] for r in rows if r["split"] == split} != set(CLASSES):
            raise ValueError("All four classes required in each development split")
    return rows


def export(root, manifest_path, output, size=192):
    if output.exists() or size < 16 or size % 16:
        raise ValueError("New output directory and input size multiple of 16 required")
    manifest = json.loads(manifest_path.read_text())
    rows = audit(manifest)
    if any(
        sum(r["split"] == split for r in rows) > 512
        for split in ("train", "validation")
    ):
        raise ValueError("Native batch maximum is 512 images per partition")
    if any(
        sum(r["split"] == split for r in rows) * 5 * size * size > 150_000_000
        for split in ("train", "validation")
    ):
        raise ValueError("Input payload exceeds native allocation limit")
    output.mkdir(parents=True)
    metadata = {
        "dataset": "FIVES",
        "classes": list(CLASSES.values()),
        "label_codes": list(CLASSES),
        "label_source": SOURCE,
        "manifest_sha256": sha256(manifest_path),
        "size": size,
        "preprocessing": "RGB/255; bilinear letterbox; no augmentation",
        "pillow_version": pillow_version,
        "patient_identity_verified": False,
        "split_unit": "existing provisional similarity groups",
        "test_evaluated": False,
        "masks_used_as_labels_or_inputs": False,
        "partitions": {},
    }
    for split, partition in [("train", 1), ("validation", 2)]:
        selected = sorted(
            (r for r in rows if r["split"] == split), key=lambda r: r["sample_id"]
        )
        batch, labels = output / (split + ".bin"), output / (split + ".tsv")
        geometries, valid_bytes = [], []
        with batch.open("xb") as f:
            f.write(
                b"OWBATCH1"
                + struct.pack("<5I", len(selected), 1, size, size, partition)
            )
            for row in selected:
                sid = row["sample_id"].encode()
                f.write(struct.pack("<I", len(sid)) + sid)
            for row in selected:
                image_path = verified(root, row["image"], row["image_sha256"])
                with Image.open(image_path) as image:
                    photo, _, valid, geometry = prepare(
                        image.convert("RGB"), [None], size
                    )
                raw = photo.tobytes()
                for channel in range(3):
                    write_float(f, (v / 255 for v in raw[channel::3]))
                valid_bytes.append(valid.tobytes())
                geometries.append(geometry)
            for _ in selected:
                write_float(f, [0.0] * (size * size))
            for valid in valid_bytes:
                write_float(f, (1.0 if x else 0.0 for x in valid))
            f.write(bytes(len(selected)))
        labels.write_text(
            "sample_id\tlabel\n"
            + "".join(
                f"{r['sample_id']}\t{list(CLASSES).index(r['disease_stratum'])}\n"
                for r in selected
            )
        )
        metadata["partitions"][split] = {
            "count": len(selected),
            "class_counts": dict(Counter(r["disease_stratum"] for r in selected)),
            "batch_sha256": sha256(batch),
            "labels_sha256": sha256(labels),
            "ids": [r["sample_id"] for r in selected],
            "image_sha256": [r["image_sha256"] for r in selected],
            "letterbox": geometries,
        }
    (output / "classification-data.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "manifest", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--size", type=int, default=192)
    a = p.parse_args()
    print(json.dumps(export(a.root, a.manifest, a.output, a.size), indent=2))
