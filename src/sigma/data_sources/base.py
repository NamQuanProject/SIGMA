"""Shared normalized schema for reflection data sources (HotpotQA, NarrativeQA, MuSiQue)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SourceExample:
    """A normalized multi-hop / long-document QA example, dataset-agnostic.

    ``context`` is a list of ``{"title": str, "sentences": list[str]}`` blocks -- the
    same shape HotpotQA already used, general enough for MuSiQue's paragraphs and a
    single-block NarrativeQA summary. Downstream code (``reflections.py``'s prompt
    builder, and everything in ``reflection_dataset.py``/``train_bootstrap.py``) only
    ever sees this shape, never a dataset's raw schema.
    """

    dataset: str
    example_id: str
    question: str
    answer: str
    context: list[dict[str, Any]]
    supporting_facts: list[dict[str, Any]] = field(default_factory=list)
    type: str | None = None
    level: str | None = None
