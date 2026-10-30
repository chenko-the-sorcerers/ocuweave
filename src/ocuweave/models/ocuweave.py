"""Research model interface. An implementation and trained weights are not yet available."""
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

@dataclass(frozen=True)
class ModelInputs:
    images: Any
    clinical_note: str | None = None
    clinical_features: Mapping[str, Any] | None = None
    # Deliberately excludes targets and ground-truth lesion masks.

@dataclass(frozen=True)
class ModelOutputs:
    predictions: Any
    visual_evidence: Any
    text_evidence: Any
    uncertainty: Any

class OcuWeave(Protocol):
    def __call__(self, inputs: ModelInputs) -> ModelOutputs:
        """Implement after the method and supervision protocol are specified."""
        ...
