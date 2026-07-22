"""MeMo-aligned reflection generation: a document-first, multi-stage pipeline (direct
facts -> indirect facts -> consolidation -> self-containment check/fix -> entity
surfacing -> cross-document synthesis), using MEMO's own prompts
(``MeMo/data_synthesis_pipeline/general_prompt_utils.py``, ported near-verbatim in
``memo_pipeline/prompts.py``) run sequentially against the OpenAI API.

This supersedes ``reflections.py``'s single-call-per-question approach for anyone who
wants closer fidelity to MEMO's actual method; ``reflections.py`` stays as the
cheaper/simpler alternative (one call per question, vs. this pipeline's roughly 4-6+
calls per *document*, since it processes documents independently of specific questions,
then only produces QA pairs at the end). Both produce the same Q_final-compatible output
schema (a `rewritten_qa` field `reflection_dataset.load_qa_examples` already knows how to
read), so everything downstream (`train_bootstrap.py`, `run_consolidation.py`) is
unaffected by which one produced a given reflections file.

Deliberately not ported from MEMO: their distributed vLLM serving, async "hedging"
(duplicate requests racing each other), and per-item checkpoint/resume machinery --
that's reliability engineering for a large-scale, self-hosted, multi-GPU setup. We run
sequentially against the OpenAI API instead, with simple **per-stage** disk caching
(each of the 6 stages below writes its result to `--cache_dir`; a rerun reuses whatever's
already there unless `--overwrite_cache`), which is enough resilience at the scale
`--limit` keeps this to.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Callable

from loguru import logger

from .data_sources import LOADERS
from .memo_pipeline.pipeline import (
    Document,
    QAPair,
    build_documents,
    combine_direct_and_indirect,
    flatten_to_records,
    group_by_doc,
    run_consolidation,
    run_crossdoc,
    run_entity_surfacing,
    run_fact_extraction,
    run_self_containment,
)
from .reflections import write_jsonl
from .utils.env import load_environment
from .utils.logging_setup import setup_logging


def _load_json(path: Path) -> Any | None:
    return json.loads(path.read_text()) if path.is_file() else None


def _save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _cached_stage(cache_dir: Path, name: str, overwrite: bool, compute: Callable[[], Any]) -> Any:
    """Run ``compute()`` and cache its (already-JSON-safe) result under
    ``cache_dir/<name>.json``, or reuse an existing cache file unless ``overwrite``.
    """

    path = cache_dir / f"{name}.json"
    if path.is_file() and not overwrite:
        logger.info(f"Reusing cached {name!r} from {path}")
        return _load_json(path)
    result = compute()
    _save_json(path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MeMo-aligned document-first, multi-stage reflection generation."
    )
    parser.add_argument("--dataset", choices=sorted(LOADERS.keys()), default="hotpotqa")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset_name", default=None, help="Override the HF repo id (--dataset hotpotqa only)")
    parser.add_argument("--config", default=None, help="HF dataset config (--dataset hotpotqa only)")
    parser.add_argument("--narrativeqa_dir", type=Path, default=None, help="Required for --dataset narrativeqa")
    parser.add_argument("--musique_path", type=Path, default=None, help="Required for --dataset musique")
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Number of *source questions* to load -- bounds the document pool size. Kept "
        "small by default: this pipeline makes many more LLM calls per example than "
        "reflections.py (roughly 4-6+ per document, not 1 per question).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--cache_dir", type=Path, default=None, help="Per-stage cache dir (default: '<output>_cache/')"
    )
    parser.add_argument("--overwrite_cache", action="store_true", help="Recompute every stage, ignoring any cache")
    parser.add_argument(
        "--min_qa_pairs", type=int, default=3, help="Minimum QA pairs a document needs before consolidation runs (MEMO default: 3)"
    )
    parser.add_argument(
        "--max_fix_attempts",
        type=int,
        default=2,
        help="Self-containment check-then-fix loop cap per QA pair (MEMO's own default is 5; "
        "capped lower here by default to bound cost -- raise it once you've seen real cost/quality numbers)",
    )
    parser.add_argument(
        "--min_docs_with_qa",
        type=int,
        default=2,
        help="Minimum documents-with-QA-pairs an evidence group needs before cross-doc synthesis runs on it",
    )
    parser.add_argument("--max_other_qa_per_batch", type=int, default=20, help="Cross-doc batch size (MEMO default: 20)")
    parser.add_argument("--skip_crossdoc", action="store_true", help="Skip stage 5 (cross-document synthesis)")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Where to write this run's log file")
    return parser.parse_args()


def main() -> None:
    load_environment()
    args = parse_args()
    setup_logging(f"memo_reflections_{args.dataset}", log_dir=args.log_dir)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required")
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    loader = LOADERS[args.dataset]
    if args.dataset == "hotpotqa":
        loader_kwargs: dict[str, Any] = dict(
            split=args.split, dataset_name=args.dataset_name, config=args.config, limit=args.limit, seed=args.seed
        )
    elif args.dataset == "narrativeqa":
        if args.narrativeqa_dir is None:
            raise ValueError(
                "--narrativeqa_dir is required for --dataset narrativeqa -- see "
                "src/sigma/data_sources/narrativeqa.py for the download link and expected layout"
            )
        loader_kwargs = dict(narrativeqa_dir=args.narrativeqa_dir, split=args.split, limit=args.limit, seed=args.seed)
    else:  # musique
        if args.musique_path is None:
            raise ValueError(
                "--musique_path is required for --dataset musique -- see "
                "src/sigma/data_sources/musique.py for the download link"
            )
        loader_kwargs = dict(musique_path=args.musique_path, limit=args.limit, seed=args.seed)

    examples = list(loader(**loader_kwargs))
    logger.info(f"Loaded {len(examples)} source examples for dataset={args.dataset!r}")
    if not examples:
        raise ValueError("No examples loaded -- check --limit/--split and the dataset path flags")

    cache_dir = args.cache_dir or (args.output.parent / f"{args.output.stem}_cache")

    # --- Stage 0: build the document pool + evidence groups ---
    doc_cache = _cached_stage(
        cache_dir,
        "documents",
        args.overwrite_cache,
        lambda: (lambda docs, groups: {"docs": {k: v.to_dict() for k, v in docs.items()}, "groups": groups})(
            *build_documents(examples)
        ),
    )
    docs = {k: Document.from_dict(v) for k, v in doc_cache["docs"].items()}
    groups: list[list[str]] = doc_cache["groups"]
    logger.info(f"{len(docs)} unique documents across {len(groups)} multi-document evidence groups")

    def qa_by_doc(name: str, compute) -> dict[str, list[QAPair]]:
        raw = _cached_stage(
            cache_dir, name, args.overwrite_cache, lambda: {k: [p.to_dict() for p in v] for k, v in compute().items()}
        )
        return {k: [QAPair.from_dict(p) for p in v] for k, v in raw.items()}

    def qa_list(name: str, compute) -> list[QAPair]:
        raw = _cached_stage(cache_dir, name, args.overwrite_cache, lambda: [p.to_dict() for p in compute()])
        return [QAPair.from_dict(p) for p in raw]

    # --- Stage 1: direct + indirect fact extraction (one call per document, each) ---
    direct = qa_by_doc("direct", lambda: run_fact_extraction(docs, client=client, model=args.model, stage="direct"))
    indirect = qa_by_doc(
        "indirect", lambda: run_fact_extraction(docs, client=client, model=args.model, stage="indirect")
    )
    combined = combine_direct_and_indirect(direct, indirect)
    logger.info(f"Direct: {sum(len(v) for v in direct.values())}, indirect: {sum(len(v) for v in indirect.values())} QA pairs")

    # --- Stage 2: consolidation (one call per document) ---
    consolidated = qa_by_doc(
        "consolidated",
        lambda: run_consolidation(docs, combined, client=client, model=args.model, min_qa_pairs=args.min_qa_pairs),
    )
    logger.info(f"Consolidated: {sum(len(v) for v in consolidated.values())} QA pairs")

    # --- Stage 3: self-containment check + fix (one[-two] call(s) per QA pair) ---
    to_check = [p for pairs in combined.values() for p in pairs] + [p for pairs in consolidated.values() for p in pairs]
    checked = qa_list(
        "checked",
        lambda: run_self_containment(docs, to_check, client=client, model=args.model, max_fix_attempts=args.max_fix_attempts),
    )
    num_self_contained = sum(1 for p in checked if p.is_self_contained)
    logger.info(f"Self-containment: {num_self_contained}/{len(checked)} passed (after up to {args.max_fix_attempts} fix attempts)")
    qa_by_doc_checked = group_by_doc(checked)

    # --- Stage 4: entity surfacing (one call per document, using only self-contained pairs) ---
    entity_surface = qa_by_doc(
        "entity_surface", lambda: run_entity_surfacing(docs, qa_by_doc_checked, client=client, model=args.model)
    )
    logger.info(f"Entity surfacing: {sum(len(v) for v in entity_surface.values())} QA pairs")

    # --- Stage 5: cross-document synthesis (optional) ---
    if args.skip_crossdoc:
        crossdoc: list[QAPair] = []
    else:
        crossdoc = qa_list(
            "crossdoc",
            lambda: run_crossdoc(
                groups,
                qa_by_doc_checked,
                client=client,
                model=args.model,
                min_docs_with_qa=args.min_docs_with_qa,
                max_other_qa_per_batch=args.max_other_qa_per_batch,
            ),
        )
        logger.info(f"Cross-document synthesis: {len(crossdoc)} QA pairs")

    records = flatten_to_records(args.dataset, docs, qa_by_doc_checked, entity_surface, crossdoc)
    write_jsonl(records, args.output)
    logger.info(f"Wrote {len(records)} reflection records to {args.output}")


if __name__ == "__main__":
    main()
