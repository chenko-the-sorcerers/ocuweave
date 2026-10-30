"""Structural audit only; it does not certify consent or absence of clinical leakage."""
import json
from pathlib import Path
from collections import Counter
from ocuweave.data.schema import ClinicalSample


def audit_manifest(path: str | Path, *, check_files: bool = False, data_root: str | Path | None = None) -> dict:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("manifest must be a non-empty JSON list")
    samples = [ClinicalSample.from_dict(row) for row in rows]
    patient_splits, seen_samples, seen_images = {}, set(), set()
    root = Path(data_root).resolve() if data_root else Path(path).resolve().parent
    for sample in samples:
        key = (sample.dataset_id, sample.patient_id)
        if key in patient_splits and patient_splits[key] != sample.split:
            raise ValueError("patient appears in multiple splits")
        patient_splits[key] = sample.split
        sample_key = (sample.dataset_id, sample.sample_id)
        if sample_key in seen_samples:
            raise ValueError("duplicate sample identifier")
        seen_samples.add(sample_key)
        for image in sample.image_paths:
            image_key = (sample.dataset_id, image)
            if image_key in seen_images:
                raise ValueError("repeated image path; review duplicates before splitting")
            seen_images.add(image_key)
        if check_files:
            for asset in (*sample.image_paths, *sample.lesion_masks):
                asset_path = (root / asset).resolve()
                if not asset_path.is_relative_to(root):
                    raise ValueError("asset path escapes data root")
                if not asset_path.is_file():
                    raise ValueError("manifest refers to a missing asset")
    return {
        "samples": len(samples), "patients": len(patient_splits),
        "splits": dict(Counter(s.split for s in samples)),
        "missing_notes": sum(not s.modality_availability["note"] for s in samples),
        "files_checked": check_files,
        "scope": "Structural checks only. Clinical leakage, patient identity, and duplicates across datasets require separate review."
    }
