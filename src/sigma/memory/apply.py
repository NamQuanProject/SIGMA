"""Attach a MemoryEntry's shared A (frozen) to a fresh backbone, and toggle a synthesized
adapter (B') on/off for inference -- the "reconstruction at inference" step (eq. 23)
applied to an actual model, plus the ability to fall back to the unmodified baseline for
comparison.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import Iterator

import torch

from ..adapters.shared_lora import SharedLoRALinear, attach_shared_lora
from .entry import MemoryEntry


def attach_memory(model, entry: MemoryEntry) -> dict[str, SharedLoRALinear]:
    """Attach shared-A LoRA wrappers to ``model`` using the entry's frozen A, zero-init B."""

    rank = next(iter(entry.shared_A.values())).shape[0]
    target_modules = [f"^{re.escape(name)}$" for name in entry.layer_bases.keys()]
    adapters = attach_shared_lora(model, rank=rank, target_modules=target_modules, seed=0)
    for name, adapter in adapters.items():
        adapter.A.copy_(entry.shared_A[name].to(adapter.A.dtype))
    return adapters


@contextmanager
def apply_adapter(
    adapters: dict[str, SharedLoRALinear], b_matrices: dict[str, torch.Tensor] | None
) -> Iterator[None]:
    """Temporarily set each adapter's B to ``b_matrices`` (or zero, i.e. baseline, if None)."""

    originals = {name: adapter.B.detach().clone() for name, adapter in adapters.items()}
    try:
        for name, adapter in adapters.items():
            value = b_matrices[name] if b_matrices is not None else torch.zeros_like(adapter.B)
            adapter.B.data.copy_(value.to(adapter.B.dtype))
        yield
    finally:
        for name, adapter in adapters.items():
            adapter.B.data.copy_(originals[name])
