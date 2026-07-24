"""MuSiQue source adapter.

Reads the **chunked corpus/questions JSONL** produced by ``process_musique.py`` -- this
is the mandatory first stage, mirroring MEMO's own two-stage pipeline
(``data_processing_utils`` -> ``data_synthesis_pipeline``). Run that script once before
calling ``load_examples`` here; a missing file raises a clear error telling you to run it.

``load_examples`` takes ``corpus_path``/``qns_path`` as two explicit file paths and
loads them the same way MEMO's own
``data_synthesis_pipeline/musique_data_utils.py`` does -- not because the algorithm is
complicated, but because the user-facing shape (CLI args, subsetting semantics, which
docs get kept) is the thing this project is trying to match exactly:

- ``corpus_path``/``qns_path`` are direct file paths, not a directory with implied
  filenames -- every ``*_datasynth_pipeline.sh`` script in MEMO takes these as two
  separate ``--corpus_path``/``--qns_path`` flags.
- ``limit`` keeps the first N questions **in file order** (``valid_count >=
  max_num_questions: break``), not a random sample -- MEMO's loaders have no shuffling
  concept at all.
- The corpus is filtered down to only the chunks referenced by the loaded questions'
  ``evidence_docs``/``gold_docs`` (``load_only_query_related_docs_musique``), rather than
  holding every chunk in the file in memory.

See ``process_musique.py`` for where the raw data comes from and what it does (chunks
each paragraph, following MEMO's own ``convert_musique_to_chunks_jsonl.py`` algorithm).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from loguru import logger

from .base import SourceExample

DATASET_LABEL = "musique"


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
                f"Missing {path} ({flag}) -- run sigma-process-musique first (the mandatory "
                f"chunking stage, mirroring MEMO's own data_processing_utils step): "
                f"sigma-process-musique --musique_path <your raw file> --output_dir <chunks dir>"
            )

    rows = _load_questions_with_evidence_docs(qns_path, limit=limit)
    corpus = _load_only_query_related_docs(corpus_path, rows)

    for row in rows:
        yield _normalize_row(row, corpus)


def _load_questions_with_evidence_docs(qns_path: Path, *, limit: int | None) -> list[dict[str, Any]]:
    """Load the first ``limit`` question rows, in file order -- mirrors
    ``load_questions_with_evidence_docs_musique`` in MEMO's ``musique_data_utils.py``
    exactly (no shuffling, no seed).
    """

    rows: list[dict[str, Any]] = []
    with qns_path.open(encoding="utf-8") as f:
        for line in f:
            if limit is not None and len(rows) >= limit:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    logger.info(f"MuSiQue: loaded {len(rows)} questions from {qns_path}")
    return rows


def _load_only_query_related_docs(corpus_path: Path, rows: list[dict[str, Any]]) -> dict[str, str]:
    """Filter the corpus down to only docids referenced by ``rows``' evidence/negative/gold
    docs -- mirrors ``load_only_query_related_docs_musique``.
    """

    query_doc_ids: set[str] = set()
    for row in rows:
        for doc in (row.get("evidence_docs") or []) + (row.get("negative_docs") or []) + (row.get("gold_docs") or []):
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

    logger.info(f"MuSiQue: loaded {len(corpus)} query-related documents from {corpus_path}")
    return corpus


def _normalize_row(row: dict[str, Any], corpus: dict[str, str]) -> SourceExample:
    evidence_docids = [d["docid"] for d in row.get("evidence_docs") or [] if d.get("docid") in corpus]
    negative_docids = [d["docid"] for d in row.get("negative_docs") or [] if d.get("docid") in corpus]

    # Context includes both supporting *and* negative (distractor) chunks -- this is
    # what gives reflection/pipeline.py's supporting-facts filtering (based on
    # supporting_facts below) something real to filter, matching the same
    # supporting/distractor distinction HotpotQA's context already has.
    context = [{"title": docid, "sentences": [corpus[docid]]} for docid in evidence_docids + negative_docids]
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
        type=row.get("hop"),
    )
