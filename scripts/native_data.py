"""Export frozen image/mask tensors for the framework-free executable.

Standard library + Pillow for image codecs/resampling only. No NumPy/PyTorch imports.
The original manifest and clinical labels remain immutable.
"""

import argparse
import array
import hashlib
import json
import io
import zipfile
import struct
import sys
from pathlib import Path
from PIL import Image, __version__ as pillow_version


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def verified(root, relative, expected):
    path = (root / relative).resolve()
    if (
        not path.is_relative_to(root.resolve())
        or not expected
        or sha256(path) != expected
    ):
        raise ValueError(
            "Missing checksum, path escape or changed file: " + str(relative)
        )
    return path


def mask_image(path):
    with Image.open(path) as source:
        if source.mode == "RGBA":
            pixels = list(source.getdata())
            if any(g or b or a != 255 for r, g, b, a in pixels):
                raise ValueError("Expected opaque red/black annotation")
            values = [r for r, _, _, _ in pixels]
        elif source.mode == "RGB":
            pixels = list(source.getdata())
            if any(r != g or r != b for r, g, b in pixels):
                raise ValueError("RGB binary masks must have identical channels")
            values = [r for r, _, _ in pixels]
        elif source.mode in {"P", "L", "1"}:
            values = list(source.getdata())
        else:
            raise ValueError("Expected indexed, scalar or red/black RGBA mask")
        if not set(values) <= {0, 1, 255}:
            raise ValueError("Unexpected mask encoding")
        result = Image.new("L", source.size)
        result.putdata([255 if v else 0 for v in values])
        return result


def load_otfid_case(root, row):
    def read_image(relative, mode):
        if root.is_file():
            with zipfile.ZipFile(root) as archive:
                if archive.namelist().count(relative) != 1:
                    raise ValueError("Missing or ambiguous archive member")
                with Image.open(io.BytesIO(archive.read(relative))) as image:
                    return image.convert(mode)
        path = (root / relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError("Archive path escapes root")
        with Image.open(path) as image:
            return image.convert(mode)

    image = read_image(row["image"], "RGB")
    pixel_hash = hashlib.sha256(
        str((image.height, image.width, 3)).encode() + image.tobytes()
    ).hexdigest()
    if pixel_hash != row["pixel_sha256"]:
        raise ValueError("OTFID decoded image checksum differs")
    union = bytearray(image.width * image.height)
    if not row["masks"] and not row.get("healthy_zero_mask"):
        raise ValueError("Missing OTFID mask is not background")
    for name in row["masks"]:
        mask = read_image(name, "L")
        if mask.size != image.size:
            raise ValueError("OTFID image-mask geometry mismatch")
        raw = mask.tobytes()
        threshold = 0 if max(raw) <= 1 else 127
        for i, value in enumerate(raw):
            union[i] |= value > threshold
    if hashlib.sha256(union).hexdigest() != row["mask_sha256"]:
        raise ValueError("OTFID decoded union-mask checksum differs")
    target = Image.frombytes("L", image.size, bytes(255 if v else 0 for v in union))
    return image, [target]


def load_case(root, row, classes):
    if isinstance(row.get("masks"), list):
        return load_otfid_case(root, row)
    path = verified(root, row["image"], row["image_sha256"])
    with Image.open(path) as source:
        image = source.convert("RGB")
    paths = row.get("masks", {classes[0]: row.get("mask")})
    hashes = row.get("mask_sha256")
    if not isinstance(hashes, dict):
        hashes = {classes[0]: hashes}
    masks = []
    for name in classes:
        mask = (
            mask_image(verified(root, paths[name], hashes[name]))
            if paths[name]
            else None
        )
        if mask is not None and mask.size != image.size:
            raise ValueError("Image-mask dimension mismatch")
        masks.append(mask)
    return image, masks


def geometry(width, height, size):
    scale = min(size / width, size / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    return w, h, (size - w) // 2, (size - h) // 2


def prepare(image, masks, size):
    w, h, x, y = geometry(*image.size, size)
    photo = Image.new("RGB", (size, size))
    photo.paste(image.resize((w, h), Image.Resampling.BILINEAR), (x, y))
    packed_masks = []
    for mask in masks:
        canvas = Image.new("L", (size, size))
        if mask is not None:
            canvas.paste(mask.resize((w, h), Image.Resampling.NEAREST), (x, y))
        packed_masks.append(canvas)
    valid = Image.new("L", (size, size))
    valid.paste(255, (x, y, x + w, y + h))
    return photo, packed_masks, valid, {"width": w, "height": h, "left": x, "top": y}


def positions(length, tile, overlap):
    if tile <= 0 or not 0 <= overlap < tile:
        raise ValueError("Invalid tile/overlap")
    if length <= tile:
        return [0]
    values = list(range(0, length - tile + 1, tile - overlap))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def tile_boxes(width, height, tile, overlap=0):
    return [
        (x, y, min(x + tile, width), min(y + tile, height))
        for y in positions(height, tile, overlap)
        for x in positions(width, tile, overlap)
    ]


def write_float(stream, values):
    data = array.array("f", values)
    if data.itemsize != 4:
        raise RuntimeError("Float32 array required")
    if sys.byteorder != "little":
        data.byteswap()
    data.tofile(stream)


def export(root, manifest_path, output, ids, size, task, patch_size=None):
    if output.exists():
        raise FileExistsError("Preserve previous export")
    if size < 16 or size % 16:
        raise ValueError("Size must be >=16 and divisible by 16")
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Unique explicit training IDs required")
    manifest = json.loads(manifest_path.read_text())
    if root.is_file() and sha256(root) != manifest.get("archive_sha256"):
        raise ValueError("Source archive checksum differs")
    classes = (
        ["MA", "HE", "EX", "SE"]
        if task == "idrid"
        else ["vessel" if task == "fives" else "lesion"]
    )
    rows = {r["sample_id"]: r for r in manifest["records"]}
    samples, provenance = [], []
    for identifier in ids:
        row = rows[identifier]
        if row["split"] != "train":
            raise ValueError("Diagnostic export accepts train only")
        image, masks = load_case(root, row, classes)
        crops = [(None, "full")]
        if patch_size:
            boxes = tile_boxes(*image.size, patch_size)
            counts = [
                (
                    sum(
                        sum(mask.crop(box).histogram()[1:])
                        for mask in masks
                        if mask is not None
                    ),
                    box,
                )
                for box in boxes
            ]
            positive = next((box for count, box in counts if count > 0), None)
            background = next((box for count, box in counts if count == 0), None)
            if positive is None or background is None:
                raise ValueError(
                    "Need both annotated-positive and zero-annotated-positive patches: "
                    + identifier
                )
            crops = [
                (positive, "positive"),
                (background, "background_observed_classes_only"),
            ]
        for box, kind in crops:
            crop_image = image.crop(box) if box else image
            crop_masks = [m.crop(box) if box and m else m for m in masks]
            photo, target, valid, transform = prepare(crop_image, crop_masks, size)
            sid = identifier if not box else identifier + "-" + kind
            samples.append((sid, photo, target, valid, [m is not None for m in masks]))
            provenance.append(
                {
                    "sample_id": sid,
                    "source_id": identifier,
                    "split": "train",
                    "original_size_wh": list(image.size),
                    "crop_xyxy": box,
                    "sampling": kind,
                    "letterbox": transform,
                    "source": row,
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(b"OWBATCH1")
        stream.write(struct.pack("<5I", len(samples), len(classes), size, size, 1))
        for sid, *_ in samples:
            encoded = sid.encode("utf-8")
            stream.write(struct.pack("<I", len(encoded)))
            stream.write(encoded)
        for _, photo, *_ in samples:
            raw = photo.tobytes()
            for channel in range(3):
                write_float(stream, (v / 255 for v in raw[channel::3]))
        for _, _, masks, _, _ in samples:
            for mask in masks:
                write_float(stream, (1.0 if v else 0.0 for v in mask.tobytes()))
        for _, _, _, valid, _ in samples:
            write_float(stream, (1.0 if v else 0.0 for v in valid.tobytes()))
        for *_, available in samples:
            stream.write(bytes(available))
    report = {
        "format": "OWBATCH1",
        "task": task,
        "classes": classes,
        "size": size,
        "manifest_sha256": sha256(manifest_path),
        "batch_sha256": sha256(output),
        "pillow_version": pillow_version,
        "samples": provenance,
        "augmentation": "none",
        "selection": "explicit IDs; optional lexical positive/background tiles",
        "background_scope": "No positive pixels in available annotations; unknown classes remain unknown",
        "patient_identity_verified": False,
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--task", choices=["idrid", "fives", "otfid"], required=True)
    p.add_argument("--patch-size", type=int)
    a = p.parse_args()
    report = export(a.root, a.manifest, a.output, a.ids, a.size, a.task, a.patch_size)
    print(
        json.dumps(
            {k: report[k] for k in ("task", "classes", "batch_sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
