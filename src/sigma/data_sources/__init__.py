"""Normalized multi-dataset sources for reflection generation: NarrativeQA, MuSiQue.
Every loader here yields the same ``SourceExample`` shape regardless of the underlying
dataset's raw schema, so ``reflections.py``, ``evaluate_sigma.py``, and everything
downstream (``reflection/dataset.py``, ``train_bootstrap.py``, ``run_consolidation.py``)
stays dataset-agnostic.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable, Iterator

from . import musique, narrativeqa
from .base import SourceExample

LOADERS: dict[str, Callable[..., Iterator[SourceExample]]] = {
    narrativeqa.DATASET_LABEL: narrativeqa.load_examples,
    musique.DATASET_LABEL: musique.load_examples,
}

__all__ = ["SourceExample", "LOADERS", "build_loader_kwargs"]


def build_loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Turn a parsed CLI namespace into ``LOADERS[args.dataset]``'s kwargs. Shared by
    ``reflections.py``, ``evaluate_sigma.py``, and ``baselines/_common.py`` so the
    `--dataset`/`--corpus_path`/`--qns_path`/`--limit` CLI surface (and its error
    messages) stays identical across every entry point. `--corpus_path`/`--qns_path` are
    two explicit file paths, matching MEMO's own
    `data_synthesis_pipeline/*_datasynth_pipeline.sh` scripts exactly -- not a directory
    with an implied filename. Both loaders need the same two paths, so there's no longer
    any per-dataset branching here now that HotpotQA (the one dataset with a different
    CLI shape -- streamed straight from Hugging Face, no local files) is gone.
    """

    if args.corpus_path is None or args.qns_path is None:
        raise ValueError(
            f"--corpus_path and --qns_path are required for --dataset {args.dataset} -- see "
            f"src/sigma/data_sources/{args.dataset}.py for the required file layout "
            f"(produced by data_process/process_{args.dataset}.py)"
        )
    return dict(corpus_path=args.corpus_path, qns_path=args.qns_path, limit=args.limit)
