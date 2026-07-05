"""Attach a MemoryEntry's shared A (frozen) to a fresh backbone, and toggle a synthesized
adapter (B') on/off for inference -- the "reconstruction at inference" step (eq. 23)
applied to an actual model, plus the ability to fall back to the unmodified baseline for
comparison.
"""

from __future__ import annotations

import re
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch

from ..adapters.shared_lora import SharedLoRALinear, attach_shared_lora
from .entry import MemoryEntry

if TYPE_CHECKING:
    from .tree import MemoryTree


def attach_memory(model, entry: MemoryEntry) -> dict[str, SharedLoRALinear]:
    """Attach shared-A LoRA wrappers to ``model`` using the entry's frozen A, zero-init B."""

    rank = next(iter(entry.shared_A.values())).shape[0]
    target_modules = [f"^{re.escape(name)}$" for name in entry.layer_bases.keys()]
    adapters = attach_shared_lora(model, rank=rank, target_modules=target_modules, seed=0)
    for name, adapter in adapters.items():
        adapter.A.copy_(entry.shared_A[name].to(adapter.A.dtype))
    return adapters


def attach_memory_tree(model, tree: "MemoryTree") -> dict[str, SharedLoRALinear]:
    """Attach shared-A LoRA wrappers sized for every task in a ``MemoryTree``.

    Routing across tasks means swapping in a *different* task's shared A for every
    candidate the tree considers (see ``apply_entry``), which only works if every leaf
    entry was bootstrapped with the same rank and target layers -- the wrapper module
    shapes are fixed at attach time and can't change per query. This checks that
    precondition up front with a clear error, then attaches using an arbitrary leaf (its
    own A/B get overwritten by ``apply_entry`` before every real use anyway).
    """

    leaves = tree.leaves()
    if not leaves:
        raise ValueError("MemoryTree has no leaves")
    reference = leaves[0].entry
    reference_layers = set(reference.layer_bases.keys())
    reference_rank = next(iter(reference.shared_A.values())).shape[0]
    for leaf in leaves[1:]:
        entry = leaf.entry
        if set(entry.layer_bases.keys()) != reference_layers:
            raise ValueError(
                f"Task {leaf.name!r} targets different layers than {leaves[0].name!r} -- "
                "every task in a MemoryTree must be bootstrapped with the same --target_modules"
            )
        rank = next(iter(entry.shared_A.values())).shape[0]
        if rank != reference_rank:
            raise ValueError(
                f"Task {leaf.name!r} has rank {rank} but {leaves[0].name!r} has rank "
                f"{reference_rank} -- every task in a MemoryTree must share --lora_rank"
            )
    return attach_memory(model, reference)


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


@contextmanager
def apply_entry(
    adapters: dict[str, SharedLoRALinear],
    entry: MemoryEntry,
    b_matrices: dict[str, torch.Tensor] | None,
) -> Iterator[None]:
    """Like ``apply_adapter``, but also swaps each layer's shared A to ``entry``'s.

    Needed for multi-task routing: unlike the single-task path (one fixed A for the
    whole run), each task in a ``MemoryTree`` was bootstrapped with its own frozen A, so
    evaluating a candidate task's context embedding requires temporarily wearing that
    task's A too, not just a synthesized B.
    """

    original_A = {name: adapter.A.detach().clone() for name, adapter in adapters.items()}
    original_B = {name: adapter.B.detach().clone() for name, adapter in adapters.items()}
    try:
        for name, adapter in adapters.items():
            adapter.A.copy_(entry.shared_A[name].to(adapter.A.dtype))
            value = b_matrices[name] if b_matrices is not None else torch.zeros_like(adapter.B)
            adapter.B.data.copy_(value.to(adapter.B.dtype))
        yield
    finally:
        for name, adapter in adapters.items():
            adapter.A.copy_(original_A[name])
            adapter.B.data.copy_(original_B[name])
