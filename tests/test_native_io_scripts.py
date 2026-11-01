import array
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from native_data import export, sha256, tile_boxes
from native_evaluate import evaluate
from native_tiles import merge_tiles


class NativeIOScripts(unittest.TestCase):
    def test_export_restore_and_missingness(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            Image.new("RGB", (16, 16), (128, 64, 32)).save(root / "image.png")
            mask = Image.new("L", (16, 16))
            mask.paste(255, (2, 3, 7, 9))
            mask.save(root / "mask.png")
            row = {
                "sample_id": "sample",
                "split": "train",
                "image": "image.png",
                "mask": "mask.png",
                "image_sha256": sha256(root / "image.png"),
                "mask_sha256": sha256(root / "mask.png"),
            }
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({"records": [row]}))
            meta = export(root, manifest, root / "input.bin", ["sample"], 16, "fives")
            self.assertEqual(
                meta["samples"][0]["letterbox"],
                {"width": 16, "height": 16, "left": 0, "top": 0},
            )
            with (root / "prob.bin").open("wb") as f:
                f.write(b"OWPROB01" + struct.pack("<4I", 1, 1, 16, 16))
                values = array.array("f", (1.0 if x else 0.0 for x in mask.tobytes()))
                if sys.byteorder != "little":
                    values.byteswap()
                values.tofile(f)
            report = evaluate(
                root / "prob.bin", root / "input.json", root, root / "result"
            )
            self.assertEqual(report["summary"]["vessel"]["positive_macro_dice"], 1)
            self.assertEqual(report["rows"][0]["predicted_area_pixels"], 30)
            self.assertEqual(
                report["rows"][0]["predicted_image_area_percent"], 100 * 30 / 256
            )
            row["split"] = "test"
            manifest.write_text(json.dumps({"records": [row]}))
            with self.assertRaises(ValueError):
                export(root, manifest, root / "forbidden.bin", ["sample"], 16, "fives")
            row["split"] = "train"
            row["mask_sha256"] = "0" * 64
            manifest.write_text(json.dumps({"records": [row]}))
            with self.assertRaises(ValueError):
                export(root, manifest, root / "changed.bin", ["sample"], 16, "fives")

    def test_overlap_reconstruction_oracle(self):
        W, H = 9, 7
        boxes = tile_boxes(W, H, 4, 2)
        tiles = [
            [float(y * W + x) for y in range(y0, y1) for x in range(x0, x1)]
            for x0, y0, x1, y1 in boxes
        ]
        self.assertEqual(
            list(merge_tiles(tiles, boxes, W, H)), list(map(float, range(W * H)))
        )
        with self.assertRaises(ValueError):
            merge_tiles(tiles[:-1], boxes[:-1], W, H)
        self.assertEqual(tile_boxes(2, 3, 4, 0), [(0, 0, 2, 3)])
