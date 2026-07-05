"""Build the reflection QA training set (Q_final) from generated HotpotQA reflections.

Loads a JSONL of reflection records (as produced by ``hotpotqa_reflections.py`` in
``--mode openai``), extracts (question, answer) pairs, and tokenizes them with the loss
masked to answer tokens only (eq. 15 in the SIGMA proposal).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

IGNORE_INDEX = -100


@dataclass(frozen=True)
class QAExample:
    """One (question, answer) pair drawn from Q_final."""

    example_id: str
    question: str
    answer: str


def load_qa_examples(reflections_path: Path) -> list[QAExample]:
    """Load Q_final from a reflections JSONL file.

    Prefers the ``rewritten_qa`` field produced by the reflection pipeline (an improved
    question/answer pair preserving the gold answer). Falls back to the original
    ``source.question`` / ``source.answer`` when ``rewritten_qa`` is missing or malformed.
    """

    examples: list[QAExample] = []
    with Path(reflections_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            examples.append(_record_to_example(record))
    return examples


def _record_to_example(record: dict[str, Any]) -> QAExample:
    source = record.get("source") or {}
    example_id = str(source.get("example_id") or record.get("id") or "")

    rewritten = record.get("rewritten_qa")
    question, answer = None, None
    if isinstance(rewritten, dict):
        question = rewritten.get("question")
        answer = rewritten.get("answer")

    if not question:
        question = source.get("question") or record.get("question")
    if not answer:
        answer = source.get("answer") or record.get("answer")

    if not question or not answer:
        raise ValueError(f"Record {example_id!r} has no usable question/answer pair")

    return QAExample(example_id=example_id, question=str(question).strip(), answer=str(answer).strip())


def build_prompt(question: str) -> str:
    """Render the training/inference prompt for a QA example."""

    return f"Question: {question}\nAnswer:"


def bootstrap_subsets(
    examples: Sequence[QAExample],
    *,
    num_subsets: int,
    subset_size: int | None = None,
    seed: int = 42,
) -> list[list[QAExample]]:
    """Draw M subsets of Q_final with replacement (bootstrap resampling).

    Mirrors the proposal's "Draw M subsets of Q_final with replacement, then train one
    adapter per subset." ``subset_size`` defaults to ``len(examples)``.
    """

    if not examples:
        raise ValueError("Cannot bootstrap from an empty example list")
    size = subset_size or len(examples)
    rng = random.Random(seed)
    return [[examples[rng.randrange(len(examples))] for _ in range(size)] for _ in range(num_subsets)]


class AnswerMaskedDataset(Dataset):
    """Tokenizes (question, answer) pairs, masking the loss to answer tokens only."""

    def __init__(self, examples: Sequence[QAExample], tokenizer, max_length: int = 512) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.examples[index]
        prompt = build_prompt(example.question)
        answer = " " + example.answer.strip()

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        bos_id = self.tokenizer.bos_token_id
        if bos_id is not None:
            # Match what tokenizer(prompt) with default add_special_tokens=True would
            # produce at inference time (evaluate_sigma.py), so train/eval prompts line up.
            prompt_ids = [bos_id] + prompt_ids
        answer_ids = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None:
            answer_ids = answer_ids + [eos_id]

        input_ids = (prompt_ids + answer_ids)[: self.max_length]
        labels = ([IGNORE_INDEX] * len(prompt_ids) + answer_ids)[: self.max_length]
        attention_mask = [1] * len(input_ids)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def collate_answer_masked(batch: Sequence[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    """Right-pad a batch of variable-length tokenized examples."""

    max_len = max(item["input_ids"].shape[0] for item in batch)

    def pad(tensor: torch.Tensor, pad_value: int) -> torch.Tensor:
        padding = max_len - tensor.shape[0]
        if padding <= 0:
            return tensor
        return torch.cat([tensor, torch.full((padding,), pad_value, dtype=tensor.dtype)])

    input_ids = torch.stack([pad(item["input_ids"], pad_token_id) for item in batch])
    labels = torch.stack([pad(item["labels"], IGNORE_INDEX) for item in batch])
    attention_mask = torch.stack([pad(item["attention_mask"], 0) for item in batch])

    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}
