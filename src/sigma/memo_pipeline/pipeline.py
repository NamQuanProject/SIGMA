"""Document-first, multi-stage reflection pipeline, mirroring MEMO's actual stage
structure and unit-of-work per stage (``MeMo/data_synthesis_pipeline/*.py``):

1. **Direct + indirect fact extraction** -- one LLM call per *document*, independent of
   any question (``run_fact_extraction``).
2. **Consolidation** -- one LLM call per document, combining that document's extracted
   QA pairs into richer multi-fact pairs (``run_consolidation``).
3. **Self-containment check + fix** -- one (or two, if a fix is needed) LLM call per
   *individual QA pair*, looped up to ``max_fix_attempts`` times (``run_self_containment``).
4. **Entity surfacing** -- one LLM call per document, using only that document's
   self-contained QA pairs (``run_entity_surfacing``).
5. **Cross-document synthesis** -- one LLM call per (anchor QA pair × batch of other
   documents' QA pairs), scoped to documents that co-occurred in the same source
   question's context (``run_crossdoc``).

Where this deliberately does *not* match MEMO: no distributed vLLM serving, no async
"hedging" (duplicate racing requests), no per-item checkpoint/resume -- see
``memo_reflections.py``'s module docstring for why. Cost-control knobs that MEMO also
has (``min_qa_pairs``, ``max_fix_attempts``, ``min_docs_with_qa``,
``max_other_qa_per_batch``) are preserved as CLI-configurable parameters here too.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

from tqdm import tqdm

from ..data_sources import SourceExample
from . import prompts
from .llm import call_llm_json

GENERATION_TEMPERATURE = 1.1
VERIFICATION_TEMPERATURE = 0.0


@dataclass
class Document:
    doc_id: str
    dataset: str
    title: str
    text: str
    date: str

    def to_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "dataset": self.dataset, "title": self.title, "text": self.text, "date": self.date}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Document":
        return cls(**d)


@dataclass
class QAPair:
    question: str
    answer: str
    stage: str
    doc_ids: list[str]
    is_self_contained: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "stage": self.stage,
            "doc_ids": self.doc_ids,
            "is_self_contained": self.is_self_contained,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QAPair":
        return cls(
            question=d["question"],
            answer=d["answer"],
            stage=d["stage"],
            doc_ids=list(d["doc_ids"]),
            is_self_contained=d.get("is_self_contained"),
            metadata=d.get("metadata") or {},
        )


def _stable_doc_id(dataset: str, title: str) -> str:
    digest = hashlib.sha1(f"{dataset}:{title}".encode("utf-8")).hexdigest()[:12]
    safe_title = "".join(ch if ch.isalnum() else "_" for ch in title)[:40].strip("_") or "doc"
    return f"{dataset}-{safe_title}-{digest}"


def _select_context_blocks(example: SourceExample) -> list[dict[str, Any]]:
    """Prefer supporting-only context blocks when we know which ones matter (HotpotQA,
    MuSiQue) -- extracting facts from (and cross-combining across) distractor paragraphs
    would be wasted cost and noise. Falls back to all context blocks when there's no
    supporting_facts signal (NarrativeQA: a single summary block anyway).
    """

    if not example.supporting_facts:
        return example.context
    supporting_titles = {str(fact.get("title")) for fact in example.supporting_facts}
    filtered = [block for block in example.context if str(block.get("title")) in supporting_titles]
    return filtered or example.context


def build_documents(examples: list[SourceExample]) -> tuple[dict[str, Document], list[list[str]]]:
    """Dedup context blocks across examples into a flat document pool, and record which
    documents co-occurred per example -- MEMO's "evidence group" for cross-doc
    combination, here just "the (supporting) context blocks of one source example."
    """

    docs: dict[str, Document] = {}
    groups: list[list[str]] = []
    for example in examples:
        group_ids: list[str] = []
        for block in _select_context_blocks(example):
            title = str(block.get("title") or "")
            text = " ".join(str(s) for s in (block.get("sentences") or [])).strip()
            if not text:
                continue
            doc_id = _stable_doc_id(example.dataset, title)
            if doc_id not in docs:
                docs[doc_id] = Document(
                    doc_id=doc_id, dataset=example.dataset, title=title, text=text,
                    date=prompts.extract_doc_metadata(text),
                )
            group_ids.append(doc_id)
        if len(group_ids) > 1:
            groups.append(group_ids)
    return docs, groups


def run_fact_extraction(
    docs: dict[str, Document], *, client: Any, model: str, stage: str, temperature: float = GENERATION_TEMPERATURE
) -> dict[str, list[QAPair]]:
    """``stage``: ``"direct"`` or ``"indirect"``."""

    prompt_fn = (
        prompts.prepare_prompt_for_direct_fact_extraction_v3
        if stage == "direct"
        else prompts.prepare_prompt_for_indirect_fact_extraction
    )
    result: dict[str, list[QAPair]] = {}
    for doc_id, doc in tqdm(list(docs.items()), desc=f"{stage} fact extraction"):
        parsed = call_llm_json(client, model=model, prompt=prompt_fn(doc.text, doc.date), temperature=temperature)
        pairs = []
        raw_pairs = parsed.get("qa_pairs") if isinstance(parsed, dict) else None
        for qa in raw_pairs or []:
            question, answer = qa.get("question"), qa.get("answer")
            if question and answer:
                pairs.append(QAPair(question=str(question).strip(), answer=str(answer).strip(), stage=stage, doc_ids=[doc_id]))
        result[doc_id] = pairs
    return result


def combine_direct_and_indirect(
    direct: dict[str, list[QAPair]], indirect: dict[str, list[QAPair]]
) -> dict[str, list[QAPair]]:
    combined: dict[str, list[QAPair]] = {}
    for doc_id in set(direct) | set(indirect):
        combined[doc_id] = list(direct.get(doc_id, [])) + list(indirect.get(doc_id, []))
    return combined


def run_consolidation(
    docs: dict[str, Document],
    qa_by_doc: dict[str, list[QAPair]],
    *,
    client: Any,
    model: str,
    min_qa_pairs: int = 3,
    temperature: float = GENERATION_TEMPERATURE,
) -> dict[str, list[QAPair]]:
    result: dict[str, list[QAPair]] = {}
    for doc_id, pairs in tqdm(list(qa_by_doc.items()), desc="consolidation"):
        if len(pairs) < min_qa_pairs:
            result[doc_id] = []
            continue
        numbered = "\n".join(f"{i + 1}. Q: {p.question}\n   A: {p.answer}" for i, p in enumerate(pairs))
        parsed = call_llm_json(
            client, model=model, prompt=prompts.prepare_prompt_for_consolidation(numbered, docs[doc_id].date),
            temperature=temperature,
        )
        consolidated = []
        raw_pairs = parsed.get("consolidated_qa_pairs") if isinstance(parsed, dict) else None
        for qa in raw_pairs or []:
            question, answer = qa.get("question"), qa.get("answer")
            if question and answer:
                consolidated.append(
                    QAPair(
                        question=str(question).strip(),
                        answer=str(answer).strip(),
                        stage="consolidated",
                        doc_ids=[doc_id],
                        metadata={
                            "combined_from": qa.get("combined_from"),
                            "commonality": qa.get("commonality"),
                            "combination_size": qa.get("combination_size"),
                        },
                    )
                )
        result[doc_id] = consolidated
    return result


def run_self_containment(
    docs: dict[str, Document],
    pairs: list[QAPair],
    *,
    client: Any,
    model: str,
    max_fix_attempts: int = 2,
    temperature: float = VERIFICATION_TEMPERATURE,
) -> list[QAPair]:
    checked: list[QAPair] = []
    for pair in tqdm(pairs, desc="self-containment check/fix"):
        doc = docs.get(pair.doc_ids[0]) if pair.doc_ids else None
        question, answer = pair.question, pair.answer
        is_ok = False
        attempts = 0
        while True:
            check = call_llm_json(
                client, model=model,
                prompt=prompts.prepare_prompt_for_self_containment_check(question, answer),
                temperature=temperature,
            )
            is_ok = bool(check.get("is_self_contained")) if isinstance(check, dict) else False
            if is_ok or attempts >= max_fix_attempts or doc is None:
                break
            fixed = call_llm_json(
                client, model=model,
                prompt=prompts.prepare_prompt_for_self_containment_fix(question, answer, doc.text, doc.date),
                temperature=temperature,
            )
            if isinstance(fixed, dict) and fixed.get("question") and fixed.get("answer"):
                question, answer = str(fixed["question"]).strip(), str(fixed["answer"]).strip()
            attempts += 1
        pair.question, pair.answer, pair.is_self_contained = question, answer, is_ok
        pair.metadata["attempts_needed"] = attempts
        checked.append(pair)
    return checked


def group_by_doc(pairs: list[QAPair]) -> dict[str, list[QAPair]]:
    grouped: dict[str, list[QAPair]] = {}
    for pair in pairs:
        for doc_id in pair.doc_ids:
            grouped.setdefault(doc_id, []).append(pair)
    return grouped


def run_entity_surfacing(
    docs: dict[str, Document],
    qa_by_doc: dict[str, list[QAPair]],
    *,
    client: Any,
    model: str,
    temperature: float = GENERATION_TEMPERATURE,
) -> dict[str, list[QAPair]]:
    result: dict[str, list[QAPair]] = {}
    for doc_id, pairs in tqdm(list(qa_by_doc.items()), desc="entity surfacing"):
        self_contained = [p for p in pairs if p.is_self_contained]
        if not self_contained:
            result[doc_id] = []
            continue
        qa_str = "\n".join(f"Q: {p.question}\nA: {p.answer}" for p in self_contained)
        parsed = call_llm_json(
            client, model=model, prompt=prompts.prepare_prompt_for_entity_surfacing(qa_str, docs[doc_id].date),
            temperature=temperature,
        )
        surfaced = []
        raw_pairs = parsed.get("entity_surfacing_qa_pairs") if isinstance(parsed, dict) else None
        for qa in raw_pairs or []:
            question, answer = qa.get("question"), qa.get("answer")
            if question and answer:
                surfaced.append(
                    QAPair(
                        question=str(question).strip(),
                        answer=str(answer).strip(),
                        stage="entity_surface",
                        doc_ids=[doc_id],
                        metadata={"entity": qa.get("entity"), "complexity": qa.get("complexity"), "facts_used": qa.get("facts_used")},
                    )
                )
        result[doc_id] = surfaced
    return result


def run_crossdoc(
    groups: list[list[str]],
    qa_by_doc: dict[str, list[QAPair]],
    *,
    client: Any,
    model: str,
    min_docs_with_qa: int = 2,
    max_other_qa_per_batch: int = 20,
    temperature: float = GENERATION_TEMPERATURE,
) -> list[QAPair]:
    results: list[QAPair] = []
    seen_questions: set[str] = set()

    for group in tqdm(groups, desc="cross-document synthesis"):
        docs_with_qa = [d for d in group if qa_by_doc.get(d)]
        if len(docs_with_qa) < min_docs_with_qa:
            continue
        for anchor_doc_id in docs_with_qa:
            other_pairs: list[tuple[str, QAPair]] = [
                (d, p) for d in docs_with_qa if d != anchor_doc_id for p in qa_by_doc[d]
            ]
            if not other_pairs:
                continue
            for anchor_qa in qa_by_doc[anchor_doc_id]:
                for batch_start in range(0, len(other_pairs), max_other_qa_per_batch):
                    batch = other_pairs[batch_start : batch_start + max_other_qa_per_batch]
                    other_qa_batch = [(d, {"question": p.question, "answer": p.answer}) for d, p in batch]
                    parsed = call_llm_json(
                        client, model=model,
                        prompt=prompts.prepare_prompt_for_crossdoc_anchor_combination(
                            {"question": anchor_qa.question, "answer": anchor_qa.answer}, anchor_doc_id, other_qa_batch
                        ),
                        temperature=temperature,
                    )
                    raw_pairs = parsed.get("crossdoc_qa_pairs") if isinstance(parsed, dict) else None
                    for qa in raw_pairs or []:
                        question, answer = qa.get("question"), qa.get("answer")
                        if not question or not answer:
                            continue
                        key = str(question).strip().lower()
                        if key in seen_questions:
                            continue
                        seen_questions.add(key)
                        results.append(
                            QAPair(
                                question=str(question).strip(),
                                answer=str(answer).strip(),
                                stage="crossdoc",
                                doc_ids=list(qa.get("source_doc_ids") or [anchor_doc_id]),
                                metadata={"type": qa.get("type")},
                            )
                        )
    return results


def flatten_to_records(
    dataset: str,
    docs: dict[str, Document],
    *stages: dict[str, list[QAPair]] | list[QAPair],
) -> list[dict[str, Any]]:
    """Flatten every stage's QA pairs into Q_final-compatible records: a top-level
    ``rewritten_qa`` (so ``reflection_dataset.load_qa_examples`` picks these up directly,
    same as the OpenAI-preferred field in the single-call pipeline) plus a ``source``
    block recording provenance (which pipeline stage produced it, which document(s)).
    """

    records: list[dict[str, Any]] = []

    def add(pair: QAPair) -> None:
        titles = [docs[d].title for d in pair.doc_ids if d in docs]
        records.append(
            {
                "rewritten_qa": {"question": pair.question, "answer": pair.answer},
                "source": {
                    "dataset": dataset,
                    "example_id": "|".join(pair.doc_ids) or pair.stage,
                    "question": pair.question,
                    "answer": pair.answer,
                    "type": pair.stage,
                    "level": None,
                    "titles": titles,
                    "is_self_contained": pair.is_self_contained,
                    "metadata": pair.metadata,
                },
            }
        )

    for stage_result in stages:
        if isinstance(stage_result, dict):
            for pairs in stage_result.values():
                for pair in pairs:
                    add(pair)
        else:
            for pair in stage_result:
                add(pair)

    return records
