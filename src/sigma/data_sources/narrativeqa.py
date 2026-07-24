"""NarrativeQA source adapter.

Reads the **chunked corpus/questions JSONL** produced by ``process_narrativeqa.py`` --
this is the mandatory first stage, mirroring MEMO's own two-stage pipeline
(``data_processing_utils`` -> ``data_synthesis_pipeline``). Run that script once per
split before calling ``load_examples`` here; a missing file raises a clear error telling
you to run it.

``load_examples`` takes ``corpus_path``/``qns_path`` as two explicit file paths and
loads them the same way MEMO's own ``data_synthesis_pipeline/nqa_data_utils.py`` does:

- ``corpus_path``/``qns_path`` are direct file paths, not a directory + ``--split`` with
  an implied filename -- MEMO's ``narrativeqa_datasynth_pipeline.sh`` takes these as two
  separate ``--corpus_path``/``--qns_path`` flags pointing at one already-chosen split's
  chunk files.
- ``limit`` keeps questions from the first N **unique source documents encountered in
  file order** (via each question's ``document_id``), not a random sample of questions --
  matching the general-purpose branch of ``load_only_query_related_docs_nqa``/
  ``load_questions_with_evidence_docs_nqa``. MEMO also special-cases exactly 3 named
  subset sizes (10 / 5_1 / 5_2 documents) via a hardcoded, pre-chosen doc-ID list
  (``nqa_subset_utils.SUBSET_MAP``) tied to *their* specific corpus build; that list isn't
  reproducible without their exact chunk IDs, so it's intentionally not ported here --
  every ``limit`` value here uses the general "first N in file order" rule instead.

See ``process_narrativeqa.py`` for where the raw data comes from and what it does
(chunks each story's Wikipedia summary, following MEMO's own
``convert_narrativeqa_to_chunks_jsonl.py`` algorithm).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from .base import SourceExample

DATASET_LABEL = "narrativeqa"


def load_examples(
    *,
    corpus_path: str | Path,
    qns_path: str | Path,
    limit: int | None = None,
    **_ignored,
) -> Iterator[SourceExample]:
    corpus_path = Path(corpus_path)
    qns_path = Path(qns_path)
    for path, flag in ((corpus_path, "--corpus_path"), (qns_path, "--qns_path")):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path} ({flag}) -- run sigma-process-narrativeqa first (the "
                f"mandatory chunking stage, mirroring MEMO's own data_processing_utils "
                f"step): sigma-process-narrativeqa --narrativeqa_dir <raw NarrativeQA "
                f"checkout> --split <split> --output_dir <chunks dir>"
            )

    rows = _load_questions_with_evidence_docs(qns_path, limit=limit)
    corpus = _load_only_query_related_docs(corpus_path, rows)

    for row in rows:
        yield _normalize_row(row, corpus)


def _load_questions_with_evidence_docs(qns_path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    """Load questions belonging to the first ``limit`` unique source documents
    encountered in file order (by ``document_id``) -- mirrors the general-purpose branch
    of ``load_questions_with_evidence_docs_nqa`` in MEMO's ``nqa_data_utils.py``.
    """

    rows: list[dict[str, Any]] = []
    seen_source_docs: set[str] = set()
    with qns_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)

            if limit is not None:
                source_doc_id = row.get("document_id")
                if source_doc_id not in seen_source_docs:
                    if len(seen_source_docs) >= limit:
                        continue
                    seen_source_docs.add(source_doc_id)

            rows.append(row)

    logger.info(
        f"NarrativeQA: loaded {len(rows)} questions from {len(seen_source_docs) or '(all)'} "
        f"source documents in {qns_path}"
    )
    return rows


def _load_only_query_related_docs(corpus_path: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Filter the corpus down to only docids referenced by ``rows``' evidence/gold docs --
    mirrors ``load_only_query_related_docs_nqa``.
    """

    query_doc_ids: set[str] = set()
    for row in rows:
        for doc in (row.get("evidence_docs") or []) + (row.get("gold_docs") or []):
            docid = doc.get("docid")
            if docid:
                query_doc_ids.add(docid)

    corpus: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            docid = doc.get("docid")
            if docid in query_doc_ids and docid not in corpus:
                corpus[docid] = doc.get("text", "")

    logger.info(f"NarrativeQA: loaded {len(corpus)} query-related documents from {corpus_path}")
    return corpus


def _normalize_row(row: dict[str, Any], corpus: dict[str, str]) -> SourceExample:
    evidence_docids = [d["docid"] for d in row.get("evidence_docs") or [] if d.get("docid") in corpus]
    context = [{"title": docid, "sentences": [corpus[docid]]} for docid in evidence_docids]
    # NarrativeQA has no distractor/negative concept -- every evidence doc is supporting.
    supporting_facts = [{"title": docid, "sent_id": 0} for docid in evidence_docids]

    answers = row.get("answers") or []
    answer = str(answers[0]).strip() if answers else ""

    return SourceExample(
        dataset=DATASET_LABEL,
        example_id=str(row.get("query_id") or ""),
        question=str(row.get("question") or "").strip(),
        answer=answer,
        context=context,
        supporting_facts=supporting_facts,
    )
