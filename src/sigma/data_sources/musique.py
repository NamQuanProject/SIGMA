"""MuSiQue source adapter.

Reads the **chunked corpus/questions JSONL** produced by ``process_musique.py`` -- this
is the mandatory first stage, mirroring MEMO's own two-stage pipeline
(``data_processing_utils`` -> ``data_synthesis_pipeline``). Run that script once before
calling ``load_examples`` here; a missing/incomplete chunked file raises a clear error
telling you to run it.

See ``process_musique.py`` for where the raw data comes from and what it does (chunks
each paragraph, following MEMO's own ``convert_musique_to_chunks_jsonl.py`` algorithm).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from .base import SourceExample

DATASET_LABEL = "musique"


def load_examples(
    *,
    musique_dir: str | Path,
    limit: int | None = None,
    seed: int = 42,
    **_ignored,
) -> Iterator[SourceExample]:
    root = Path(musique_dir)
    corpus_path = root / "musique_corpus_chunks.jsonl"
    questions_path = root / "musique_questions_chunks.jsonl"
    for path in (corpus_path, questions_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path} -- run process_musique.py first (the mandatory chunking "
                f"stage, mirroring MEMO's own data_processing_utils step): "
                f"python process_musique.py --musique_path <your raw file> --output_dir {musique_dir}"
            )

    corpus: dict[str, str] = {}
    with corpus_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            corpus[row["docid"]] = row["text"]

    rows: list[dict[str, Any]] = []
    with questions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)

    for row in rows:
        yield _normalize_row(row, corpus)


def _normalize_row(row: dict[str, Any], corpus: dict[str, str]) -> SourceExample:
    evidence_docids = [d["docid"] for d in row.get("evidence_docs") or [] if d.get("docid") in corpus]
    negative_docids = [d["docid"] for d in row.get("negative_docs") or [] if d.get("docid") in corpus]

    # Context includes both supporting *and* negative (distractor) chunks -- this is
    # what gives reflection_pipeline.py's supporting-facts filtering (based on
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
