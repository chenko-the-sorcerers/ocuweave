"""Bounded, pickle-free integrity audit of the official FairVLMed arrays."""
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np

from ocuweave.data.auditing.clinical_notes import audit_notes
from ocuweave.utils.provenance import sha256, write_json


MAX_NPZ_BYTES = 64 * 1024 * 1024
MAX_ARRAY_BYTES = 32 * 1024 * 1024


def safe_array(inner, name):
    """Validate header before allocation; never deserialize Python objects."""
    info = inner.getinfo(name)
    if info.file_size > MAX_ARRAY_BYTES:
        raise ValueError("Array exceeds audit allocation limit")
    with inner.open(info) as stream:
        version = np.lib.format.read_magic(stream)
        readers = {(1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0}
        if version not in readers:
            raise ValueError("Unsupported NPY version")
        shape, _, dtype = readers[version](stream)
        if dtype.hasobject or dtype.kind not in "buif":
            raise ValueError("Only primitive numeric arrays may be decoded")
        if math.prod(shape) * dtype.itemsize > MAX_ARRAY_BYTES:
            raise ValueError("Declared array exceeds audit allocation limit")
    with inner.open(info) as stream:
        return np.load(io.BytesIO(stream.read()), allow_pickle=False)


def inspect_npz(payload):
    if len(payload) > MAX_NPZ_BYTES:
        raise ValueError("NPZ exceeds audit allocation limit")
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        names = inner.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Ambiguous repeated array names")
        if "slo_fundus.npy" not in names:
            raise ValueError("Missing official slo_fundus array")
        image = safe_array(inner, "slo_fundus.npy")
        label = safe_array(inner, "glaucoma.npy")
    if image.ndim not in {2, 3} or min(image.shape[:2]) < 16:
        raise ValueError("Unsupported image dimensions")
    if image.ndim == 3 and image.shape[-1] not in {1, 3}:
        raise ValueError("Unsupported image channels")
    if not np.isfinite(image).all():
        raise ValueError("Non-finite image pixels")
    if label.size != 1 or float(label.item()) not in {0, 1}:
        raise ValueError("Invalid glaucoma label")
    h = hashlib.sha256()
    h.update(str(image.shape).encode()); h.update(image.dtype.str.encode())
    h.update(np.ascontiguousarray(image).tobytes())
    result = {"shape": list(image.shape), "dtype": str(image.dtype),
              "min": float(image.min()), "max": float(image.max()),
              "mean": float(image.mean()), "std": float(image.std()),
              "constant": bool(image.min() == image.max()),
              "pixel_sha256": h.hexdigest(), "npz_label": int(label.item())}
    return result


def audit_images(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("Preserve prior audit; choose a new output")
    table, archive = root / "ReadMe/data_summary.csv", root / "Dataset/dataset.zip"
    with table.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"filename", "use", "note", "glaucoma"}.issubset(rows[0]):
        raise ValueError("Unexpected clinical table schema")
    if len({r["filename"] for r in rows}) != len(rows):
        raise ValueError("Duplicate table filenames")
    if any(r["use"] not in {"training", "validation", "test"} or
           r["glaucoma"] not in {"yes", "no"} for r in rows):
        raise ValueError("Unexpected split or label")
    hashes = {"archive_sha256": sha256(archive), "csv_sha256": sha256(table)}
    # Compare acquired bytes with the existing receipt, when available.
    receipt = root / "download_manifest.json"
    if receipt.exists():
        expected = {item["path"]: item["sha256"] for item in json.loads(receipt.read_text())["files"]}
        if expected.get("Dataset/dataset.zip") != hashes["archive_sha256"] or expected.get("ReadMe/data_summary.csv") != hashes["csv_sha256"]:
            raise ValueError("Dataset bytes differ from acquisition receipt")
        hashes["acquisition_receipt_sha256"] = sha256(receipt)
    output.mkdir(parents=True)
    records, failures, groups = [], [], defaultdict(list)
    with zipfile.ZipFile(archive) as source:
        members = [n for n in source.namelist() if n.endswith(".npz") and "__MACOSX" not in n]
        if any(PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts for n in members):
            raise ValueError("Unsafe member path")
        index = {PurePosixPath(n).name: n for n in members}
        if len(index) != len(members):
            raise ValueError("Ambiguous archive filenames")
        for number, row in enumerate(rows, 1):
            identity = {"filename": row["filename"], "split": row["use"]}
            try:
                member = index[row["filename"]]
                if source.getinfo(member).file_size > MAX_NPZ_BYTES:
                    raise ValueError("NPZ exceeds audit allocation limit")
                stats = inspect_npz(source.read(member))
                stats["label_matches_csv"] = stats.pop("npz_label") == (row["glaucoma"] == "yes")
                record = {**identity, **stats}
                records.append(record)
                groups[stats["pixel_sha256"]].append(identity)
            except (KeyError, ValueError, OSError, EOFError, zipfile.BadZipFile):
                # Exception text can contain arbitrary archive data; log only type-neutral failure.
                failures.append({**identity, "reason": "numeric_image_or_label_integrity_failure"})
            if number % 500 == 0:
                write_json(output / "progress.json", {"processed": number, "total": len(rows), "failures": len(failures)})
                print(json.dumps({"processed": number, "total": len(rows), "failures": len(failures)}), flush=True)
    duplicate_groups = [g for g in groups.values() if len(g) > 1]
    summary = {"protocol": "fairvlmed-numeric-integrity-v1", **hashes,
               "rows": len(rows), "decoded": len(records), "failed": len(failures),
               "unreferenced_npz": len(set(index) - {r["filename"] for r in rows}),
               "shapes": dict(Counter(str(r["shape"]) for r in records)),
               "dtypes": dict(Counter(r["dtype"] for r in records)),
               "constant_images": sum(r["constant"] for r in records),
               "label_mismatches": sum(not r["label_matches_csv"] for r in records),
               "exact_pixel_duplicate_groups": len(duplicate_groups),
               "cross_split_exact_pixel_groups": sum(len({r["split"] for r in g}) > 1 for g in duplicate_groups),
               "official_split_counts": dict(Counter(r["use"] for r in rows)),
               "allow_pickle": False, "raw_notes_logged": False,
               "patient_linkage_independently_verified": False,
               "clinical_image_quality_adjudicated": False,
               "limitations": ["Numeric integrity is not clinical gradability or de-identification review.",
                               "Exact hashes do not detect near duplicates or prove patient independence.",
                               "No records are deleted, relabeled, or reassigned between splits."]}
    write_json(output / "image_records.json", records)
    write_json(output / "image_failures.json", failures)
    write_json(output / "duplicate_groups.json", duplicate_groups)
    write_json(output / "note_review.json", audit_notes(rows))
    write_json(output / "summary.json", summary)
    return summary
