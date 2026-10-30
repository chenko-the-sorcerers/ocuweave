"""Patient-linked manifest contract. Targets are kept separate from model inputs."""
from dataclasses import dataclass, field
from typing import Any, Mapping

@dataclass(frozen=True)
class ClinicalSample:
    dataset_id: str
    sample_id: str
    patient_id: str
    split: str
    image_paths: tuple[str, ...]
    laterality: str | None = None
    visit_id: str | None = None
    clinical_note: str | None = None
    clinical_features: Mapping[str, Any] = field(default_factory=dict)
    targets: Mapping[str, Any] = field(default_factory=dict)
    lesion_masks: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for name in ("dataset_id", "sample_id", "patient_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a non-empty string; do not invent patient IDs")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("split must be train, validation, or test")
        if self.laterality not in {None, "left", "right", "unknown"}:
            raise ValueError("invalid laterality")
        if not isinstance(self.image_paths, (tuple, list)) or not self.image_paths:
            raise ValueError("at least one image path is required")
        if any(not isinstance(p, str) or not p.strip() for p in self.image_paths):
            raise ValueError("image paths must be non-empty strings")
        if self.clinical_note is not None and not isinstance(self.clinical_note, str):
            raise ValueError("clinical_note must be a string or null")
        if self.visit_id is not None and not isinstance(self.visit_id, str):
            raise ValueError("visit_id must be a string or null")
        for name in ("clinical_features", "targets", "provenance"):
            if not isinstance(getattr(self, name), Mapping):
                raise ValueError(f"{name} must be an object")
        if not isinstance(self.lesion_masks, (tuple, list)) or any(not isinstance(p, str) or not p for p in self.lesion_masks):
            raise ValueError("lesion_masks must contain path strings")

    @property
    def modality_availability(self) -> dict[str, bool]:
        return {"image": bool(self.image_paths), "note": bool(self.clinical_note and self.clinical_note.strip()),
                "clinical": bool(self.clinical_features)}

    @classmethod
    def from_dict(cls, value: dict) -> "ClinicalSample":
        if not isinstance(value, dict):
            raise ValueError("each sample must be an object")
        return cls(**value)
