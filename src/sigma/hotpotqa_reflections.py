"""Generate MEMO-style QA reflections from the HotpotQA dataset.

This module loads HotpotQA from Hugging Face, normalizes the examples, and
builds structured prompts that ask an LLM to produce reflection data suitable
for SIGMA's bootstrap-and-consolidate pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from datasets import load_dataset
from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency guard
    load_dotenv = None


DEFAULT_DATASET_NAME = "hotpot_qa"
DEFAULT_DATASET_CONFIGS = ("distractor", "fullwiki")


@dataclass(frozen=True)
class HotpotQAExample:
    """Normalized HotpotQA record."""

    example_id: str
    question: str
    answer: str
    supporting_facts: list[dict[str, Any]]
    context: list[dict[str, Any]]
    type: str | None = None
    level: str | None = None


def load_hotpotqa_examples(
    *,
    split: str = "train",
    config: str | None = None,
    streaming: bool = False,
    limit: int | None = None,
    seed: int = 42,
) -> Iterator[HotpotQAExample]:
    """Yield normalized HotpotQA examples.

    The Hugging Face dataset is commonly exposed under the `hotpot_qa` name with
    the `distractor` and `fullwiki` configurations. If `config` is not supplied,
    we try those in order.
    """

    configs = (config,) if config else DEFAULT_DATASET_CONFIGS
    dataset = None
    last_error: Exception | None = None
    for candidate in configs:
        try:
            dataset = load_dataset(DEFAULT_DATASET_NAME, candidate, split=split, streaming=streaming)
            break
        except Exception as exc:  # pragma: no cover - fallback path
            last_error = exc
    if dataset is None:
        raise RuntimeError(
            f"Could not load {DEFAULT_DATASET_NAME!r} with configs {configs!r}"
        ) from last_error

    if streaming:
        iterable: Iterable[dict[str, Any]] = dataset
        if limit is not None:
            iterable = _take(iterable, limit)
    else:
        rows = list(dataset)
        if limit is not None and limit < len(rows):
            rng = random.Random(seed)
            rows = rng.sample(rows, limit)
        iterable = rows

    for row in iterable:
        yield normalize_hotpotqa_row(row)


def normalize_hotpotqa_row(row: dict[str, Any]) -> HotpotQAExample:
    """Normalize a raw HotpotQA row into a stable schema."""

    example_id = str(row.get("_id") or row.get("id") or row.get("qid") or "")
    if not example_id:
        example_id = f"hotpotqa-{abs(hash(json.dumps(row, sort_keys=True, default=str)))}"

    supporting_facts_raw = row.get("supporting_facts") or []
    supporting_facts: list[dict[str, Any]] = []
    if isinstance(supporting_facts_raw, dict):
        supporting_facts = [
            {"title": title, "sent_id": sent_id}
            for title, sent_ids in supporting_facts_raw.items()
            for sent_id in (sent_ids if isinstance(sent_ids, list) else [sent_ids])
        ]
    else:
        for fact in supporting_facts_raw:
            if isinstance(fact, dict):
                supporting_facts.append(
                    {
                        "title": fact.get("title"),
                        "sent_id": fact.get("sent_id"),
                    }
                )
            elif isinstance(fact, (list, tuple)) and len(fact) >= 2:
                supporting_facts.append({"title": fact[0], "sent_id": fact[1]})

    context_raw = row.get("context") or []
    context: list[dict[str, Any]] = []
    for item in context_raw:
        if isinstance(item, dict):
            context.append(
                {
                    "title": item.get("title"),
                    "sentences": item.get("sentences") or [],
                }
            )
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            context.append({"title": item[0], "sentences": item[1]})

    return HotpotQAExample(
        example_id=example_id,
        question=str(row.get("question", "")).strip(),
        answer=str(row.get("answer", "")).strip(),
        supporting_facts=supporting_facts,
        context=context,
        type=row.get("type"),
        level=row.get("level"),
    )


def build_reflection_prompt(example: HotpotQAExample) -> str:
    """Create a structured prompt for QA reflection generation."""

    context_lines = []
    for item in example.context:
        title = item.get("title", "")
        sentences = item.get("sentences") or []
        context_lines.append(f"[{title}] {' '.join(str(s) for s in sentences)}")

    supporting_facts = json.dumps(example.supporting_facts, ensure_ascii=False)
    context_block = "\n".join(context_lines) if context_lines else "(no context provided)"

    return f"""You are generating reflection data for a multi-hop QA memory system.

Given the HotpotQA example below, produce a JSON object with these keys:
- fact_extraction: short bullet list of relevant facts from the context
- reasoning_reflection: concise explanation of how the facts connect
- answer_verification: whether the provided answer is supported by the evidence
- entity_surface: list of key entities that matter for retrieval
- cross_document_synthesis: concise synthesis across supporting documents
- rewritten_qa: an improved question and answer pair, preserving the gold answer

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
    example: HotpotQAExample,
    dataset_config: str | None = None,
    temperature: float = 0.2,
) -> dict[str, Any]:
    """Generate reflection JSON for a single example using an OpenAI client."""

    prompt = build_reflection_prompt(example)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {
                "role": "system",
                "content": "You generate strict JSON for training data.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    content = response.choices[0].message.content or "{}"
    parsed = _parse_json_object(content)
    parsed["source"] = {
        "dataset": DEFAULT_DATASET_NAME,
        "config": dataset_config,
        "example_id": example.example_id,
        "question": example.question,
        "answer": example.answer,
        "supporting_facts": example.supporting_facts,
    }
    return parsed


def generate_reflection_record(example: HotpotQAExample) -> dict[str, Any]:
    """Build a prompt record without calling an LLM.

    This is useful for offline prompt export or debugging the dataset pipeline.
    """

    return {
        "id": example.example_id,
        "question": example.question,
        "answer": example.answer,
        "supporting_facts": example.supporting_facts,
        "context": example.context,
        "prompt": build_reflection_prompt(example),
    }


def write_jsonl(records: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    _load_environment()

    parser = argparse.ArgumentParser(description="Generate QA reflections from HotpotQA.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--config", default=None, help="hotpot_qa config, usually distractor or fullwiki")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of examples to process")
    parser.add_argument("--streaming", action="store_true", help="Use streaming dataset access")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        choices=("prompt", "openai"),
        default="prompt",
        help="Export prompts or generate reflections with OpenAI",
    )
    parser.add_argument("--model", default=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    examples = load_hotpotqa_examples(
        split=args.split,
        config=args.config,
        streaming=args.streaming,
        limit=args.limit,
        seed=args.seed,
    )

    if args.mode == "prompt":
        records = (generate_reflection_record(example) for example in tqdm(examples, desc="exporting prompts"))
        write_jsonl(records, args.output)
        return

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for --mode openai")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    def _generated_records() -> Iterator[dict[str, Any]]:
        for example in tqdm(examples, desc="generating reflections"):
            yield generate_reflection_with_openai(
                client=client,
                model=args.model,
                example=example,
                dataset_config=args.config,
            )

    write_jsonl(_generated_records(), args.output)


def _parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model output, tolerating code fences."""

    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].lstrip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        stripped = stripped[start : end + 1]
    return json.loads(stripped)


def _take(iterable: Iterable[Any], limit: int) -> Iterator[Any]:
    for index, item in enumerate(iterable):
        if index >= limit:
            break
        yield item


def _load_environment() -> None:
    """Load environment variables from local .env files if available."""

    if load_dotenv is None:
        return

    repo_root = Path(__file__).resolve().parents[2]
    env_candidates = [
        repo_root / ".env",
        repo_root / ".env.local",
        repo_root / ".env" / ".env",
        repo_root / ".env" / "local.env",
    ]
    for candidate in env_candidates:
        if candidate.is_file():
            load_dotenv(candidate, override=False)


if __name__ == "__main__":
    main()
