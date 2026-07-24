"""Normalized multi-dataset sources for reflection generation: HotpotQA, NarrativeQA,
MuSiQue. Every loader here yields the same ``SourceExample`` shape regardless of the
underlying dataset's raw schema, so ``reflections.py``, ``evaluate_sigma.py``, and
everything downstream (``reflection_dataset.py``, ``train_bootstrap.py``,
``run_consolidation.py``) stays dataset-agnostic.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Iterator

from . import hotpotqa, musique, narrativeqa
from .base import SourceExample

LOADERS: dict[str, Callable[..., Iterator[SourceExample]]] = {
    hotpotqa.DATASET_LABEL: hotpotqa.load_examples,
    narrativeqa.DATASET_LABEL: narrativeqa.load_examples,
    musique.DATASET_LABEL: musique.load_examples,
}

__all__ = ["SourceExample", "LOADERS", "build_loader_kwargs"]


def build_loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Turn a parsed CLI namespace into ``LOADERS[args.dataset]``'s kwargs. Shared by
    ``reflections.py`` and ``evaluate_sigma.py`` so the `--dataset`/`--narrativeqa_dir`/
    `--musique_dir`/`--split` CLI surface (and its "which file layout does each dataset
    need" error messages) stays identical across both entry points.
    """

    if args.dataset == "hotpotqa":
        return dict(
            split=args.split,
            dataset_name=args.dataset_name,
            config=args.config,
            streaming=args.streaming,
            limit=args.limit,
            seed=args.seed,
        )
    if args.dataset == "narrativeqa":
        if args.narrativeqa_dir is None:
            raise ValueError(
                "--narrativeqa_dir is required for --dataset narrativeqa -- see "
                "src/sigma/data_sources/narrativeqa.py for the required file layout "
                "(produced by process_narrativeqa.py)"
            )
        return dict(narrativeqa_dir=args.narrativeqa_dir, split=args.split, limit=args.limit, seed=args.seed)
    # musique
    if args.musique_dir is None:
        raise ValueError(
            "--musique_dir is required for --dataset musique -- see "
            "src/sigma/data_sources/musique.py for the required file layout "
            "(produced by process_musique.py)"
        )
    return dict(musique_dir=args.musique_dir, limit=args.limit, seed=args.seed)
