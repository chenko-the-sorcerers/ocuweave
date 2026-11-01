"""Framework-free tiled inference and uniform overlap reconstruction.

Python standard library handles geometry/accumulation; Pillow handles images.
Every original pixel must receive at least one prediction. No training labels
are used to select inference tiles.
"""

import argparse
import array
import json
import struct
import subprocess
import tempfile
from pathlib import Path
from PIL import Image
from native_data import prepare, tile_boxes, write_float, sha256
from native_evaluate import read_probabilities


def merge_tiles(tiles, boxes, width, height):
    total = array.array("f", [0.0]) * (width * height)
    counts = array.array("I", [0]) * (width * height)
    if len(tiles) != len(boxes):
        raise ValueError("Tile/box count mismatch")
    for values, (x0, y0, x1, y1) in zip(tiles, boxes):
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height) or len(values) != (
            x1 - x0
        ) * (y1 - y0):
            raise ValueError("Tile geometry mismatch")
        for y in range(y0, y1):
            for x in range(x0, x1):
                index = y * width + x
                total[index] += values[(y - y0) * (x1 - x0) + x - x0]
                counts[index] += 1
    if not counts or min(counts) == 0:
        raise ValueError("Uncovered original image pixels")
    return array.array("f", (v / n for v, n in zip(total, counts)))


def infer(
    executable,
    checkpoint,
    source,
    output,
    classes,
    size=384,
    tile_size=384,
    overlap=96,
    device="cpu",
):
    if output.exists() or classes < 1 or size < 16 or size % 16:
        raise ValueError("New output directory and valid class/input sizes required")
    with Image.open(source) as img:
        image = img.convert("RGB")
    boxes = tile_boxes(*image.size, tile_size, overlap)
    tiles = [[] for _ in range(classes)]
    with tempfile.TemporaryDirectory(prefix="ocuweave-native-tiles-") as folder:
        folder = Path(folder)
        for index, box in enumerate(boxes):
            photo, targets, valid, geometry = prepare(
                image.crop(box), [None] * classes, size
            )
            batch = folder / "input.bin"
            prediction = folder / f"prediction-{index}.bin"
            with batch.open("wb") as stream:
                stream.write(
                    b"OWBATCH1" + struct.pack("<5I", 1, classes, size, size, 2)
                )
                sid = f"tile-{index}".encode()
                stream.write(struct.pack("<I", len(sid)) + sid)
                rgb = photo.tobytes()
                for c in range(3):
                    write_float(stream, (v / 255 for v in rgb[c::3]))
                for _ in range(classes):
                    write_float(stream, [0.0] * (size * size))
                write_float(stream, (1.0 if v else 0.0 for v in valid.tobytes()))
                stream.write(bytes(classes))
            subprocess.run(
                [str(executable), str(checkpoint), str(batch), str(prediction), device],
                check=True,
                capture_output=True,
            )
            shape, values = read_probabilities(prediction)
            if shape != (1, classes, size, size):
                raise ValueError("Unexpected native prediction shape")
            for c in range(classes):
                p = Image.new("F", (size, size))
                p.putdata(values[c * size * size : (c + 1) * size * size])
                g = geometry
                p = p.crop(
                    (
                        g["left"],
                        g["top"],
                        g["left"] + g["width"],
                        g["top"] + g["height"],
                    )
                )
                p = p.resize(
                    (box[2] - box[0], box[3] - box[1]), Image.Resampling.BILINEAR
                )
                tiles[c].append(array.array("f", p.getdata()))
            prediction.unlink()
    output.mkdir(parents=True)
    for c in range(classes):
        merged = merge_tiles(tiles[c], boxes, *image.size)
        p = Image.new("F", image.size)
        p.putdata(merged)
        p.save(output / f"class-{c}-probability.tiff")
        mask = Image.new("L", image.size)
        mask.putdata([255 if v >= 0.5 else 0 for v in merged])
        mask.save(output / f"class-{c}-mask.png")
    report = {
        "scope": "tiled inference; no accuracy claim",
        "source_sha256": sha256(source),
        "checkpoint_sha256": sha256(checkpoint),
        "image_size_wh": list(image.size),
        "input_size": size,
        "tile_size": tile_size,
        "overlap": overlap,
        "boxes_xyxy": boxes,
        "merge": "uniform probability average before threshold",
        "threshold": 0.5,
        "device": device,
        "early_detection_validated": False,
    }
    (output / "tiling.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--executable", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--image", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--classes", type=int, required=True)
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--tile-size", type=int, default=384)
    p.add_argument("--overlap", type=int, default=96)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    a = p.parse_args()
    r = infer(
        a.executable,
        a.checkpoint,
        a.image,
        a.output,
        a.classes,
        a.size,
        a.tile_size,
        a.overlap,
        a.device,
    )
    print(json.dumps({"tiles": len(r["boxes_xyxy"]), "output": str(a.output)}))
