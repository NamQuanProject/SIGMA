"""Normalized multi-dataset sources for reflection generation: HotpotQA, NarrativeQA,
MuSiQue. Every loader here yields the same ``SourceExample`` shape regardless of the
underlying dataset's raw schema, so ``reflections.py`` and everything downstream of it
(``reflection_dataset.py``, ``train_bootstrap.py``, ``run_consolidation.py``) stays
dataset-agnostic.
"""

from __future__ import annotations

from typing import Callable, Iterator

from . import hotpotqa, musique, narrativeqa
from .base import SourceExample

LOADERS: dict[str, Callable[..., Iterator[SourceExample]]] = {
    hotpotqa.DATASET_LABEL: hotpotqa.load_examples,
    narrativeqa.DATASET_LABEL: narrativeqa.load_examples,
    musique.DATASET_LABEL: musique.load_examples,
}

__all__ = ["SourceExample", "LOADERS"]
