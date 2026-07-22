"""Generate QA reflections from any of the supported source datasets (HotpotQA,
NarrativeQA, MuSiQue -- ``data_sources/LOADERS``).

This generalizes ``hotpotqa_reflections.py`` (kept as-is, unchanged, since
``evaluate_sigma.py`` still uses its ``load_hotpotqa_examples`` for the HotpotQA-only
eval path) to any dataset in ``data_sources/``, all normalized to the same
``SourceExample`` shape -- so the output JSONL schema, and everything downstream that
consumes it (``reflection_dataset.py``, ``train_bootstrap.py``, ``run_consolidation.py``),
is identical regardless of which source dataset a given reflections file came from.

HotpotQA loads from Hugging Face; NarrativeQA and MuSiQue load from **local files**
instead (neither is reliably published on Hugging Face) -- see
``data_sources/narrativeqa.py`` and ``data_sources/musique.py`` for exact download links
and expected file layout, or pass ``--dataset narrativeqa``/``--dataset musique`` with no
path flag to get the same instructions as an error message.

The reflection prompt asks for six fields in one LLM call per example -- fact
extraction (direct facts stated outright, plus indirect facts that require combining two
or more sentences), a reasoning reflection (how the facts connect), answer verification,
entity surfacing ("describe the entity without naming it"), cross-document synthesis
(what combining more than one context block tells you), and a rewritten, self-contained
(question, answer) pair. This is our own reflection-generation approach, not a port of
MEMO's actual multi-stage/multi-call synthesis pipeline (fact extraction split across two
separate calls, an iterative self-containment check/fix loop, etc.) -- one call per
example keeps this simple and cheap.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Iterator

from loguru import logger
from tqdm import tqdm

from .data_sources import LOADERS, SourceExample
from .hotpotqa_reflections import _parse_json_object, write_jsonl
from .utils.env import load_environment
from .utils.logging_setup import setup_logging


def build_reflection_prompt(example: SourceExample) -> str:
    """Create a structured prompt for QA reflection generation."""

    context_lines = []
    for item in example.context:
        title = item.get("title", "")
        sentences = item.get("sentences") or []
        context_lines.append(f"[{title}] {' '.join(str(s) for s in sentences)}")

    supporting_facts = json.dumps(example.supporting_facts, ensure_ascii=False)
    context_block = "\n".join(context_lines) if context_lines else "(no context provided)"

    return f"""You are generating reflection data for a multi-hop / long-document QA memory system, \
following a five-step synthesis process (fact extraction, consolidation, verification, entity \
surfacing, cross-document synthesis).

Given the {example.dataset} example below, produce a JSON object with these keys:
- fact_extraction: short bullet list of explicit, self-contained facts stated directly in the \
context (direct facts), plus any facts that require combining two or more sentences to derive \
(indirect facts -- e.g. computing an age, resolving a pronoun, chaining a cause to its effect)
- reasoning_reflection: concise explanation of how the extracted facts connect/combine to \
support the answer (this is the "consolidation" step)
- answer_verification: whether the provided answer is directly supported by the evidence, and \
if not, what's missing
- entity_surface: a short description of the key entity/entities the question is really about, \
written the way you'd describe it without naming it (its distinguishing facts), plus the entity's \
actual name
- cross_document_synthesis: concise synthesis across the different context blocks/documents -- \
what does combining information from more than one of them tell you that no single one does alone?
- rewritten_qa: an improved, self-contained question and answer pair that preserves the gold \
answer exactly

Rules:
- Keep the output valid JSON.
- Do not include chain-of-thought style private reasoning.
- Keep every field concise but useful for later training.

Example id: {example.example_id}
Question: {example.question}
Gold answer: {example.answer}
Supporting facts: {supporting_facts}
Context:
{context_block}
"""


def generate_reflection_with_openai(
    *,
    client: Any,
    model: str,
    example: SourceExample,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Generate reflection JSON for a single example using an OpenAI client."""

    prompt = build_reflection_prompt(example)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You generate strict JSON for training data."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = _parse_json_object(content)
    parsed["source"] = {
        "dataset": example.dataset,
        "example_id": example.example_id,
        "question": example.question,
        "answer": example.answer,
        "supporting_facts": example.supporting_facts,
        "type": example.type,
        "level": example.level,
    }
    return parsed


def generate_reflection_record(example: SourceExample) -> dict[str, Any]:
    """Build a prompt record without calling an LLM (offline prompt export/debugging)."""

    return {
        "id": example.example_id,
        "dataset": example.dataset,
        "question": example.question,
        "answer": example.answer,
        "supporting_facts": example.supporting_facts,
        "context": example.context,
        "type": example.type,
        "level": example.level,
        "prompt": build_reflection_prompt(example),
    }


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
        help="Local checkout of https://github.com/google-deepmind/narrativeqa (needs documents.csv, "
        "qaps.csv, third_party/wikipedia/summaries.csv). Required for --dataset narrativeqa.",
    )
    parser.add_argument(
        "--musique_path",
        type=Path,
        default=None,
        help="Local MuSiQue JSON/JSONL file (see https://github.com/StonyBrookNLP/musique for the "
        "download link). Required for --dataset musique.",
    )
    parser.add_argument("--limit", type=int, default=100, help="Maximum number of examples to process")
    parser.add_argument(
        "--streaming", action="store_true", help="Use streaming dataset access (--dataset hotpotqa only)"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("prompt", "openai"),
        default="prompt",
        help="Export prompts or generate reflections with OpenAI",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--log_dir", type=Path, default=Path("logs"), help="Where to write this run's log file")
    args = parser.parse_args()

    setup_logging(f"reflections_{args.dataset}", log_dir=args.log_dir)

    loader = LOADERS[args.dataset]
    if args.dataset == "hotpotqa":
        loader_kwargs: dict[str, Any] = dict(
            split=args.split,
            dataset_name=args.dataset_name,
            config=args.config,
            streaming=args.streaming,
            limit=args.limit,
            seed=args.seed,
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

    examples: Iterator[SourceExample] = loader(**loader_kwargs)

    if args.mode == "prompt":
        records = (generate_reflection_record(example) for example in tqdm(examples, desc="exporting prompts"))
        write_jsonl(records, args.output)
        logger.info(f"Wrote prompt records to {args.output}")
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --mode openai")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    logger.info(f"Generating reflections for dataset={args.dataset!r} with model={args.model!r}")

    def _generated_records() -> Iterator[dict[str, Any]]:
        for example in tqdm(examples, desc="generating reflections"):
            yield generate_reflection_with_openai(client=client, model=args.model, example=example)

    write_jsonl(_generated_records(), args.output)
    logger.info(f"Wrote reflections to {args.output}")


if __name__ == "__main__":
    main()
