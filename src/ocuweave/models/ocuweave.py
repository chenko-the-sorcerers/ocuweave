"""Experimental evidence-conditioned LQNet composition; synthetic verification only."""

from dataclasses import dataclass
from typing import Any, Mapping
import torch
from torch import nn
from ocuweave.models.local_detail import LocalDetailSegmenter
from ocuweave.models.language.notes import NoteEncoder
from ocuweave.models.clinical.structured import ClinicalEncoder
from ocuweave.models.fusion.evidence import EvidenceFusion


@dataclass(frozen=True)
class ModelInputs:
    images: Any
    clinical_note: str | None = None
    clinical_features: Mapping[str, Any] | None = None
    note_ids: Any = None
    note_valid: Any = None
    clinical_values: Any = None
    clinical_observed: Any = None
    # Targets, reference masks and outcome labels are intentionally absent.


@dataclass(frozen=True)
class ModelOutputs:
    predictions: Any
    visual_evidence: Any
    text_evidence: Any
    uncertainty: Any


class OcuWeave(nn.Module):
    """Tensor-level research composition, not a clinically trained multimodal system.

    Call tokenize_notes explicitly and supply feature values in a fixed documented
    order. ClinicalEncoder.fit_statistics accepts training rows only. No dataset
    pairing, note clearance, diagnosis, or faithful explanation is inferred here.
    """

    def __init__(
        self, clinical_features=3, width=24, dim=48, queries=8, max_note_length=256
    ):
        super().__init__()
        self.vision = LocalDetailSegmenter(width, dim, queries)
        self.language = NoteEncoder(dim, max_note_length)
        self.clinical = ClinicalEncoder(clinical_features, dim)
        self.fusion = EvidenceFusion(dim)

    def forward(self, inputs):
        if not isinstance(inputs, ModelInputs):
            raise ValueError("Pass ModelInputs with explicit availability")
        if inputs.clinical_note is not None or inputs.clinical_features is not None:
            raise ValueError(
                "Encode raw notes with tokenize_notes and provide ordered clinical tensors explicitly"
            )
        images = inputs.images
        b = images.shape[0]
        dev = images.device
        if (inputs.note_ids is None) != (inputs.note_valid is None) or (
            inputs.clinical_values is None
        ) != (inputs.clinical_observed is None):
            raise ValueError("Every modality requires its explicit availability mask")
        ids = (
            inputs.note_ids
            if inputs.note_ids is not None
            else torch.zeros((b, 1), dtype=torch.long, device=dev)
        )
        valid = (
            inputs.note_valid
            if inputs.note_valid is not None
            else torch.zeros((b, 1), dtype=torch.bool, device=dev)
        )
        values = (
            inputs.clinical_values
            if inputs.clinical_values is not None
            else images.new_zeros((b, self.clinical.features))
        )
        observed = (
            inputs.clinical_observed
            if inputs.clinical_observed is not None
            else torch.zeros_like(values, dtype=torch.bool)
        )
        if ids.shape[0] != b or values.shape[0] != b:
            raise ValueError("Modality batches must match the image batch")
        notes, note_available = self.language(ids, valid)
        clinical, clinical_available = self.clinical(values, observed)
        evidence, available = self.fusion(
            torch.stack((notes, clinical), dim=1),
            torch.stack((note_available, clinical_available), dim=1),
        )
        logits = self.vision(images, evidence, available)
        p = logits.sigmoid().clamp(1e-7, 1 - 1e-7)
        entropy = -(p * p.log() + (1 - p) * (1 - p).log())
        return ModelOutputs(
            logits,
            {"mask_logits": logits},
            {
                "tokens": evidence,
                "available": available,
                "interpretation": "conditioning features, not faithful clinical explanations",
            },
            {"binary_entropy": entropy, "calibrated": False},
        )
