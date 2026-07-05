"""Trivial single-task stand-in for SIGMA's cross-task memory tree (``tree.py``,
section 4.2.2).

With a single task, routing is a no-op: every query goes to the one consolidated entry.
This exposes the same ``route(context_fn) -> (MemoryEntry, Tensor)`` shape as
``MemoryTree.route``, so ``evaluate_sigma.py`` doesn't need an "if tree else" branch for
the routing/synthesis logic -- only for which of these two classes it constructs.
"""

from __future__ import annotations

from typing import Callable

import torch

from .entry import MemoryEntry


class SingleEntryMemory:
    def __init__(self, entry: MemoryEntry) -> None:
        self.entry = entry

    def route(self, context_fn: Callable[[MemoryEntry], torch.Tensor]) -> tuple[MemoryEntry, torch.Tensor]:
        return self.entry, context_fn(self.entry)
