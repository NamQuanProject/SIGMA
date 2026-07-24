"""MEMO-aligned, document-first reflection synthesis pipeline.

Ported from the deleted ``memo_pipeline/pipeline.py``, and wired into
``reflections.py``'s ``--mode openai`` path so it's the actual reflection generator this
project ships, not a side experiment.

MEMO's own pipeline works **document-first**: it extracts facts from each document once
(not once per question), consolidates/verifies/self-contains them, then optionally
surfaces named entities and synthesizes across documents that co-occurred in the same
example. This mirrors that shape:

1. ``build_documents``      -- dedup every context block across all loaded examples into
                                one ``Document`` per unique (dataset, title); remember
                                which documents co-occurred in which example (``groups``)
                                so cross-document synthesis only combines documents that
                                actually appeared together, not arbitrary pairs.
2. ``run_fact_extraction``  -- per document: direct + indirect fact extraction (two LLM
                                calls, MEMO's own two-prompt split).
3. ``run_consolidation``    -- per document: combine related facts into richer QA pairs.
4. ``run_self_containment`` -- per QA pair (every stage so far): check, and fix if it
                                fails, up to ``max_fix_retries`` attempts (kept low
                                deliberately -- this is O(#qa_pairs) LLM calls already).
5. ``run_entity_surfacing`` -- per document: "describe the entity without naming it"
                                QA pairs.
6. ``run_crossdoc``         -- per co-occurrence group with >=2 documents: one
                                anchor-vs-batch combination call (not the full pairwise
                                cross product -- see its docstring for why).

``flatten_to_records`` converts the resulting per-document QA pairs into the same
``source``/``rewritten_qa`` record schema ``dataset.py`` already expects, so
nothing downstream (``train_bootstrap.py``, ``run_consolidation.py``) needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from loguru import logger
from tqdm import tqdm

from ..data_sources.base import SourceExample
from .llm import call_llm_json
from .prompts import (
    extract_doc_metadata,
    prepare_prompt_for_consolidation,
    prepare_prompt_for_crossdoc_anchor_combination,
    prepare_prompt_for_direct_fact_extraction_v3,
    prepare_prompt_for_entity_surfacing,
    prepare_prompt_for_indirect_fact_extraction,
    prepare_prompt_for_self_containment_check,
    prepare_prompt_for_self_containment_fix,
)


@dataclass
class QAPair:
    question: str
    answer: str
    source_stage: str
    doc_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "source_stage": self.source_stage,
            "doc_ids": list(self.doc_ids),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QAPair":
        return cls(
            question=str(data.get("question", "")),
            answer=str(data.get("answer", "")),
            source_stage=str(data.get("source_stage", "")),
            doc_ids=list(data.get("doc_ids") or []),
        )


@dataclass
class Document:
    doc_id: str
    dataset: str
    text: str
    date: str
    example_ids: list[str] = field(default_factory=list)
    qa_pairs: list[QAPair] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "dataset": self.dataset,
            "text": self.text,
            "date": self.date,
            "example_ids": list(self.example_ids),
            "qa_pairs": [qa.to_dict() for qa in self.qa_pairs],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Document":
        return cls(
            doc_id=str(data["doc_id"]),
            dataset=str(data.get("dataset", "")),
            text=str(data.get("text", "")),
            date=str(data.get("date", "")),
            example_ids=list(data.get("example_ids") or []),
            qa_pairs=[QAPair.from_dict(qa) for qa in data.get("qa_pairs") or []],
        )


def _stable_doc_id(dataset: str, title: str) -> str:
    return f"{dataset}::{title}"


def _select_context_blocks(example: SourceExample) -> list[dict[str, Any]]:
    """Prefer supporting-only context blocks (matched by title against
    ``supporting_facts``); fall back to every context block when there are no
    supporting facts to match against, or none of them match (so we never silently
    drop the only context an example has).
    """

    supporting_titles = {sf.get("title") for sf in (example.supporting_facts or [])}
    if supporting_titles:
        matched = [block for block in example.context if block.get("title") in supporting_titles]
        if matched:
            return matched
    return list(example.context)


def build_documents(examples: Iterable[SourceExample]) -> tuple[dict[str, Document], list[list[str]]]:
    """Dedup context blocks into unique ``Document``s, and record per-example
    co-occurrence groups (the doc_ids selected for that one example) for later use by
    ``run_crossdoc``.
    """

    documents: dict[str, Document] = {}
    groups: list[list[str]] = []

    for example in examples:
        blocks = _select_context_blocks(example)
        group_doc_ids: list[str] = []

        for block in blocks:
            title = str(block.get("title", ""))
            text = " ".join(str(s) for s in (block.get("sentences") or [])).strip()
            if not text:
                continue
            doc_id = _stable_doc_id(example.dataset, title)
            if doc_id not in documents:
                documents[doc_id] = Document(
                    doc_id=doc_id,
                    dataset=example.dataset,
                    text=text,
                    date=extract_doc_metadata(text),
                    example_ids=[example.example_id],
                )
            else:
                doc = documents[doc_id]
                if text not in doc.text:
                    doc.text = f"{doc.text}\n{text}"
                if example.example_id not in doc.example_ids:
                    doc.example_ids.append(example.example_id)
            if doc_id not in group_doc_ids:
                group_doc_ids.append(doc_id)

        if len(group_doc_ids) >= 2:
            groups.append(group_doc_ids)

    return documents, groups


def combine_direct_and_indirect(
    direct_pairs: list[dict[str, Any]], indirect_pairs: list[dict[str, Any]], doc_id: str
) -> list[QAPair]:
    combined = []
    for raw in direct_pairs:
        if raw.get("question") and raw.get("answer"):
            combined.append(QAPair(str(raw["question"]), str(raw["answer"]), "direct", [doc_id]))
    for raw in indirect_pairs:
        if raw.get("question") and raw.get("answer"):
            combined.append(QAPair(str(raw["question"]), str(raw["answer"]), "indirect", [doc_id]))
    return combined


def run_fact_extraction(client: Any, *, model: str, documents: dict[str, Document]) -> None:
    for doc in tqdm(documents.values(), desc="fact extraction", total=len(documents)):
        direct = call_llm_json(
            client, model=model, prompt=prepare_prompt_for_direct_fact_extraction_v3(doc.text, doc.date)
        )
        indirect = call_llm_json(
            client, model=model, prompt=prepare_prompt_for_indirect_fact_extraction(doc.text, doc.date)
        )
        direct_pairs = direct.get("qa_pairs") if isinstance(direct, dict) else None
        indirect_pairs = indirect.get("qa_pairs") if isinstance(indirect, dict) else None
        doc.qa_pairs.extend(
            combine_direct_and_indirect(direct_pairs or [], indirect_pairs or [], doc.doc_id)
        )
        if not doc.qa_pairs:
            logger.warning(f"No facts extracted for document {doc.doc_id!r}")


def run_consolidation(client: Any, *, model: str, documents: dict[str, Document]) -> None:
    import json

    for doc in tqdm(documents.values(), desc="consolidation", total=len(documents)):
        if len(doc.qa_pairs) < 2:
            continue
        indexed = [{"index": i + 1, **qa.to_dict()} for i, qa in enumerate(doc.qa_pairs)]
        result = call_llm_json(
            client,
            model=model,
            prompt=prepare_prompt_for_consolidation(json.dumps(indexed, ensure_ascii=False), doc.date),
        )
        consolidated = result.get("consolidated_qa_pairs") if isinstance(result, dict) else None
        for raw in consolidated or []:
            if raw.get("question") and raw.get("answer"):
                doc.qa_pairs.append(QAPair(str(raw["question"]), str(raw["answer"]), "consolidated", [doc.doc_id]))


def run_self_containment(
    client: Any, *, model: str, documents: dict[str, Document], max_fix_retries: int = 1
) -> None:
    """Check every QA pair generated so far and repair it if it fails, up to
    ``max_fix_retries`` attempts each. Kept low deliberately: this stage is already
    O(#qa_pairs) LLM calls, and MEMO's own self-containment loop is one of the more
    expensive parts of the pipeline to run for real.
    """

    for doc in tqdm(documents.values(), desc="self-containment", total=len(documents)):
        for qa in doc.qa_pairs:
            for _ in range(max_fix_retries):
                check = call_llm_json(
                    client, model=model, prompt=prepare_prompt_for_self_containment_check(qa.question, qa.answer)
                )
                if isinstance(check, dict) and check.get("is_self_contained"):
                    break
                fixed = call_llm_json(
                    client,
                    model=model,
                    prompt=prepare_prompt_for_self_containment_fix(qa.question, qa.answer, doc.text, doc.date),
                )
                if isinstance(fixed, dict) and fixed.get("question") and fixed.get("answer"):
                    qa.question = str(fixed["question"])
                    qa.answer = str(fixed["answer"])
                else:
                    break


def group_by_doc(documents: dict[str, Document]) -> dict[str, list[QAPair]]:
    return {doc_id: list(doc.qa_pairs) for doc_id, doc in documents.items()}


def run_entity_surfacing(client: Any, *, model: str, documents: dict[str, Document]) -> None:
    import json

    for doc in tqdm(documents.values(), desc="entity surfacing", total=len(documents)):
        if not doc.qa_pairs:
            continue
        qa_pairs_str = json.dumps([qa.to_dict() for qa in doc.qa_pairs], ensure_ascii=False)
        result = call_llm_json(
            client, model=model, prompt=prepare_prompt_for_entity_surfacing(qa_pairs_str, doc.date)
        )
        surfaced = result.get("entity_surfacing_qa_pairs") if isinstance(result, dict) else None
        for raw in surfaced or []:
            if raw.get("question") and raw.get("answer"):
                doc.qa_pairs.append(
                    QAPair(str(raw["question"]), str(raw["answer"]), "entity_surfacing", [doc.doc_id])
                )


def run_crossdoc(
    client: Any,
    *,
    model: str,
    documents: dict[str, Document],
    groups: list[list[str]],
    batch_size: int = 8,
) -> list[QAPair]:
    """For each co-occurrence group of >=2 documents, run a single anchor-vs-batch
    cross-document combination call (anchor = the first QA pair of the group's first
    document with any QA pairs; batch = up to ``batch_size`` QA pairs drawn round-robin
    from the group's other documents).

    This is intentionally **not** the full pairwise/combinatorial cross product MEMO's
    real pipeline can run at large scale -- one call per group keeps cost linear in the
    number of examples rather than quadratic in the number of documents, which matters
    since this pipeline runs sequentially against a single OpenAI-compatible client.
    """

    crossdoc_pairs: list[QAPair] = []

    for group in tqdm(groups, desc="cross-document synthesis"):
        docs_in_group = [documents[doc_id] for doc_id in group if doc_id in documents]
        anchor_doc = next((d for d in docs_in_group if d.qa_pairs), None)
        if anchor_doc is None:
            continue
        anchor_qa = anchor_doc.qa_pairs[0]

        other_docs = [d for d in docs_in_group if d.doc_id != anchor_doc.doc_id]
        others: list[tuple[str, dict[str, Any]]] = []
        max_per_doc = max((len(d.qa_pairs) for d in other_docs), default=0)
        for pair_idx in range(max_per_doc):
            for doc in other_docs:
                if pair_idx < len(doc.qa_pairs):
                    others.append((doc.doc_id, doc.qa_pairs[pair_idx].to_dict()))
                if len(others) >= batch_size:
                    break
            if len(others) >= batch_size:
                break

        if not others:
            continue

        result = call_llm_json(
            client,
            model=model,
            prompt=prepare_prompt_for_crossdoc_anchor_combination(anchor_qa.to_dict(), anchor_doc.doc_id, others),
        )
        found = result.get("crossdoc_qa_pairs") if isinstance(result, dict) else None
        for raw in found or []:
            if raw.get("question") and raw.get("answer"):
                source_doc_ids = raw.get("source_doc_ids") or group
                crossdoc_pairs.append(
                    QAPair(str(raw["question"]), str(raw["answer"]), "crossdoc", list(source_doc_ids))
                )

    return crossdoc_pairs


def flatten_to_records(
    documents: dict[str, Document], crossdoc_pairs: list[QAPair], *, dataset: str
) -> list[dict[str, Any]]:
    """Flatten every stage's QA pairs into the ``source``/``rewritten_qa`` record shape
    ``reflection_dataset.load_qa_examples`` already knows how to read -- ``source.type``
    is repurposed to carry the pipeline stage that produced the record (direct, indirect,
    consolidated, entity_surfacing, crossdoc), which is also usable as a
    ``--question_type``-style filter if a caller wants only one stage's output.
    """

    records: list[dict[str, Any]] = []

    def _emit(qa: QAPair, doc_ids: list[str]) -> None:
        example_id = f"{'+'.join(doc_ids)}:{qa.source_stage}:{len(records)}"
        records.append(
            {
                "source": {
                    "dataset": dataset,
                    "example_id": example_id,
                    "question": qa.question,
                    "answer": qa.answer,
                    "supporting_facts": None,
                    "type": qa.source_stage,
                    "level": None,
                },
                "rewritten_qa": {"question": qa.question, "answer": qa.answer},
                "doc_ids": doc_ids,
            }
        )

    for doc in documents.values():
        for qa in doc.qa_pairs:
            _emit(qa, qa.doc_ids)
    for qa in crossdoc_pairs:
        _emit(qa, qa.doc_ids)

    return records
