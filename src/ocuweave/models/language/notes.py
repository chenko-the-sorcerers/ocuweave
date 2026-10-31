"""Small random-initialized byte Transformer; not a pretrained or generative LLM."""

import torch
from torch import nn


def tokenize_notes(notes, max_length=256):
    if type(max_length) is not int or max_length < 1 or not notes:
        raise ValueError("Nonempty batch and positive token length required")
    ids = torch.zeros((len(notes), max_length), dtype=torch.long)
    for i, note in enumerate(notes):
        if note is not None and not isinstance(note, str):
            raise ValueError("Notes must be strings or None")
        if note and note.strip():
            values = [byte + 1 for byte in note.encode("utf-8")[:max_length]]
            ids[i, : len(values)] = torch.tensor(values, dtype=torch.long)
    return ids, ids.ne(0)


class NoteEncoder(nn.Module):
    def __init__(self, dim=48, max_length=256, layers=2):
        super().__init__()
        if dim < 4 or dim % 4 or max_length < 1 or layers < 1:
            raise ValueError("Invalid text encoder dimensions")
        self.max_length = max_length
        self.embedding = nn.Embedding(257, dim, padding_idx=0)
        self.position = nn.Parameter(torch.randn(max_length, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(
            dim, 4, dim * 2, dropout=0, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)

    def forward(self, ids, valid):
        if (
            ids.ndim != 2
            or valid.shape != ids.shape
            or valid.dtype != torch.bool
            or ids.dtype != torch.long
            or not 0 < ids.shape[1] <= self.max_length
        ):
            raise ValueError("Expected BxT integer IDs and boolean token availability")
        safe_ids = torch.where(valid, ids, torch.zeros_like(ids))
        if ((safe_ids < 0) | (safe_ids > 256)).any():
            raise ValueError("Byte token IDs must be in [0,256]")
        available = valid.any(1)
        safe_valid = valid.clone()
        safe_valid[~available, 0] = True
        x = self.embedding(safe_ids) + self.position[: ids.shape[1]][None]
        hidden = self.encoder(x, src_key_padding_mask=~safe_valid)
        pooled = (hidden * valid[..., None]).sum(1) / valid.sum(1).clamp_min(1)[:, None]
        return (
            torch.where(available[:, None], pooled, torch.zeros_like(pooled)),
            available,
        )
