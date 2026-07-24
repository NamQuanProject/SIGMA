"""Normalized multi-dataset sources for reflection generation: HotpotQA, NarrativeQA,
MuSiQue. Every loader here yields the same ``SourceExample`` shape regardless of the
underlying dataset's raw schema, so ``reflections.py``, ``evaluate_sigma.py``, and
everything downstream (``reflection/dataset.py``, ``train_bootstrap.py``,
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
    ``reflections.py``, ``evaluate_sigma.py``, and ``baselines/_common.py`` so the
    `--dataset`/`--corpus_path`/`--qns_path`/`--split` CLI surface (and its "which file
    layout does each dataset need" error messages) stays identical across every entry
    point. `--corpus_path`/`--qns_path` (narrativeqa/musique only) are two explicit file
    paths, matching MEMO's own `data_synthesis_pipeline/*_datasynth_pipeline.sh` scripts
    exactly -- not a directory with an implied filename.
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
        if args.corpus_path is None or args.qns_path is None:
            raise ValueError(
                "--corpus_path and --qns_path are required for --dataset narrativeqa -- "
                "see src/sigma/data_sources/narrativeqa.py for the required file layout "
                "(produced by sigma-process-narrativeqa)"
            )
        return dict(corpus_path=args.corpus_path, qns_path=args.qns_path, limit=args.limit)
    # musique
    if args.corpus_path is None or args.qns_path is None:
        raise ValueError(
            "--corpus_path and --qns_path are required for --dataset musique -- see "
            "src/sigma/data_sources/musique.py for the required file layout "
            "(produced by sigma-process-musique)"
        )
    return dict(corpus_path=args.corpus_path, qns_path=args.qns_path, limit=args.limit)
