"""Trivial stand-in for SIGMA's cross-task Gromov-Wasserstein memory tree (section 4.2.2).

With a single task (HotpotQA), routing is a no-op: every query goes to the one
consolidated entry. This class exposes the same ``route(context_embedding) -> MemoryEntry``
interface a future ``MemoryTree`` would implement, so downstream inference code
(``apply.py`` / ``evaluate_sigma.py``) doesn't need to change when a real tree is built.
"""

from __future__ import annotations

import torch

from .entry import MemoryEntry


class SingleEntryMemory:
    def __init__(self, entry: MemoryEntry) -> None:
        self.entry = entry

    def route(self, context_embedding: torch.Tensor) -> MemoryEntry:
        del context_embedding  # unused: only one entry exists
        return self.entry
