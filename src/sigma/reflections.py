"""Generate QA reflections from any of the supported source datasets (HotpotQA,
NarrativeQA, MuSiQue -- ``data_sources/LOADERS``).

This generalizes ``hotpotqa_reflections.py`` (kept as-is, unchanged, since
``evaluate_sigma.py`` still uses its ``load_hotpotqa_examples`` for the HotpotQA-only
eval path) to any dataset in ``data_sources/``, all normalized to the same
``SourceExample`` shape -- so the output JSONL schema, and everything downstream that
consumes it (``reflection_dataset.py``, ``train_bootstrap.py``, ``run_consolidation.py``),
is identical regardless of which source dataset a given reflections file came from.

HotpotQA loads from Hugging Face; NarrativeQA and MuSiQue require a **local, already
chunked** corpus -- run ``process_narrativeqa.py``/``process_musique.py`` once first (see
``data_sources/narrativeqa.py``/``data_sources/musique.py``).

``--mode openai`` and ``--mode hf`` both run the real, MEMO-aligned reflection pipeline
(``reflection_pipeline.py``): document-first fact extraction (direct + indirect),
consolidation, a self-containment check/fix loop, entity surfacing, and cross-document
synthesis -- not one LLM call per question. They differ only in which client
``reflection_pipeline.py`` is handed: ``openai`` calls the OpenAI API; ``hf`` loads a
local, open-source instruction-tuned model (e.g. Qwen2.5-Instruct) via
``reflection_hf_client.HFChatClient``, which exposes the same
``chat.completions.create(...)`` shape, so the pipeline code itself doesn't change at
all between the two. ``--mode prompt`` is a cheap, offline dry-run that only exports the
stage-1 (direct fact extraction) prompt per document, for inspecting token
counts/coverage before spending real compute either way.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Iterator

from loguru import logger
from tqdm import tqdm

from .data_sources import LOADERS, SourceExample
from .hotpotqa_reflections import write_jsonl
from .reflection_pipeline import (
    build_documents,
    flatten_to_records,
    run_consolidation,
    run_crossdoc,
    run_entity_surfacing,
    run_fact_extraction,
    run_self_containment,
)
from .reflection_prompts import prepare_prompt_for_direct_fact_extraction_v3
from .utils.env import load_environment
from .utils.logging_setup import setup_logging

DEFAULT_HF_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def build_loader_kwargs(args: argparse.Namespace) -> dict[str, Any]:
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


def export_stage1_prompts(examples: Iterator[SourceExample]) -> Iterator[dict[str, Any]]:
    """Cheap, offline dry-run: one record per unique document with its stage-1 (direct
    fact extraction) prompt, for estimating cost/coverage without calling an LLM. Does
    not run consolidation/self-containment/entity-surfacing/cross-doc -- those only
    exist once stage 1's actual output is available.
    """

    documents, _groups = build_documents(examples)
    for doc in documents.values():
        yield {
            "doc_id": doc.doc_id,
            "dataset": doc.dataset,
            "example_ids": doc.example_ids,
            "date": doc.date,
            "prompt": prepare_prompt_for_direct_fact_extraction_v3(doc.text, doc.date),
        }


def run_pipeline(
    examples: Iterator[SourceExample],
    *,
    client: Any,
    model: str,
    dataset: str,
    max_fix_retries: int = 1,
) -> list[dict[str, Any]]:
    """Run the full MEMO-aligned pipeline end to end and return flattened output
    records ready to write to JSONL.
    """

    documents, groups = build_documents(tqdm(examples, desc="loading examples"))
    logger.info(f"Built {len(documents)} unique documents from {len(groups)} co-occurrence groups")

    run_fact_extraction(client, model=model, documents=documents)
    run_consolidation(client, model=model, documents=documents)
    run_self_containment(client, model=model, documents=documents, max_fix_retries=max_fix_retries)
    run_entity_surfacing(client, model=model, documents=documents)
    crossdoc_pairs = run_crossdoc(client, model=model, documents=documents, groups=groups)

    records = flatten_to_records(documents, crossdoc_pairs, dataset=dataset)
    logger.info(f"Flattened to {len(records)} reflection records")
    return records


def main() -> None:
    load_environment()

    parser = argparse.ArgumentParser(
        description="Generate QA reflections from a source dataset (HotpotQA, NarrativeQA, MuSiQue)."
    )
    parser.add_argument("--dataset", choices=sorted(LOADERS.keys()), default="hotpotqa")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--dataset_name", default=None, help="Override the HF dataset repo id for the chosen --dataset"
    )
    parser.add_argument(
        "--config", default=None, help="HF dataset config, e.g. distractor/fullwiki (--dataset hotpotqa only)"
    )
    parser.add_argument(
        "--narrativeqa_dir",
        type=Path,
        default=None,
        help="Directory containing narrativeqa_<split>_corpus_chunks.jsonl / "
        "_questions_chunks.jsonl (produced by process_narrativeqa.py). Required for "
        "--dataset narrativeqa.",
    )
    parser.add_argument(
        "--musique_dir",
        type=Path,
        default=None,
        help="Directory containing musique_corpus_chunks.jsonl / musique_questions_chunks.jsonl "
        "(produced by process_musique.py). Required for --dataset musique.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of examples to process")
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming dataset access (--dataset hotpotqa only)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("prompt", "openai", "hf"),
        default="prompt",
        help="prompt: cheap offline stage-1-prompt export; openai: run the full pipeline "
        "against the OpenAI API; hf: run the full pipeline against a local, open-source "
        "instruction-tuned model (e.g. Qwen2.5-Instruct)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="openai: chat-completions model name (default: $OPENAI_MODEL or gpt-4.1-mini). "
        f"hf: local/HF Hub model name or path (default: {DEFAULT_HF_MODEL}).",
    )
    parser.add_argument(
        "--max_fix_retries",
        type=int,
        default=1,
        help="Max self-containment fix attempts per QA pair (kept low: this stage is "
        "already O(#qa_pairs) LLM calls)",
    )
    parser.add_argument(
        "--device", default=None, help="--mode hf only: torch device (default: cuda if available, else cpu)"
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "fp32", "fp16", "bf16"],
        default="auto",
        help="--mode hf only: auto = bf16 on CUDA, fp32 on CPU",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=4096,
        help="--mode hf only: max tokens to generate per call (reflection prompts ask for "
        "JSON with potentially many QA pairs, so this is generous by default)",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Where to write this run's log file")
    args = parser.parse_args()

    setup_logging(f"reflections_{args.dataset}", log_dir=args.log_dir)

    loader = LOADERS[args.dataset]
    loader_kwargs = build_loader_kwargs(args)
    examples: Iterator[SourceExample] = loader(**loader_kwargs)

    if args.mode == "prompt":
        write_jsonl(export_stage1_prompts(examples), args.output)
        logger.info(f"Wrote stage-1 prompt records to {args.output}")
        return

    if args.mode == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --mode openai")

        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        model = args.model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    else:  # hf
        import torch

        from .reflection_hf_client import HFChatClient

        model = args.model or DEFAULT_HF_MODEL
        device = torch.device(args.device) if args.device else None
        if args.dtype == "auto":
            dtype = None  # HFChatClient resolves auto (bf16 on CUDA, fp32 on CPU) itself
        else:
            dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
        client = HFChatClient(model, device=device, dtype=dtype, max_new_tokens=args.max_new_tokens)

    logger.info(f"Generating reflections for dataset={args.dataset!r} with mode={args.mode!r} model={model!r}")

    records = run_pipeline(
        examples,
        client=client,
        model=model,
        dataset=args.dataset,
        max_fix_retries=args.max_fix_retries,
    )
    write_jsonl(records, args.output)
    logger.info(f"Wrote reflections to {args.output}")


if __name__ == "__main__":
    main()
