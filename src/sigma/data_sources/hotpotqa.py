"""HotpotQA source adapter -- thin wrapper over ``reflection/hotpotqa_legacy.py``'s existing,
already-working loader (including its HF-repo-migration fallback), just normalized to
the shared ``SourceExample`` schema every source in this package exposes.
"""

from __future__ import annotations

from typing import Iterator

from ..reflection.hotpotqa_legacy import load_hotpotqa_examples
from .base import SourceExample

DATASET_LABEL = "hotpotqa"


def load_examples(
    *,
    split: str = "train",
    dataset_name: str | None = None,
    config: str | None = None,
    streaming: bool = False,
    limit: int | None = None,
    seed: int = 42,
) -> Iterator[SourceExample]:
    for example in load_hotpotqa_examples(
        split=split, dataset_name=dataset_name, config=config, streaming=streaming, limit=limit, seed=seed
    ):
        yield SourceExample(
            dataset=DATASET_LABEL,
            example_id=example.example_id,
            question=example.question,
            answer=example.answer,
            context=example.context,
            supporting_facts=example.supporting_facts,
            type=example.type,
            level=example.level,
        )
