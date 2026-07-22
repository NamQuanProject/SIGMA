"""NarrativeQA source adapter.

Reads the **chunked corpus/questions JSONL** produced by ``process_narrativeqa.py`` --
this is the mandatory first stage, mirroring MEMO's own two-stage pipeline
(``data_processing_utils`` -> ``data_synthesis_pipeline``). Run that script once per
split before calling ``load_examples`` here; a missing/incomplete chunked file raises a
clear error telling you to run it.

See ``process_narrativeqa.py`` for where the raw data comes from and what it does
(chunks each story's Wikipedia summary, following MEMO's own
``convert_narrativeqa_to_chunks_jsonl.py`` algorithm).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

from .base import SourceExample

DATASET_LABEL = "narrativeqa"


def load_examples(
    *,
    narrativeqa_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
    **_ignored,
) -> Iterator[SourceExample]:
    root = Path(narrativeqa_dir)
    corpus_path = root / f"narrativeqa_{split}_corpus_chunks.jsonl"
    questions_path = root / f"narrativeqa_{split}_questions_chunks.jsonl"
    for path in (corpus_path, questions_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path} -- run process_narrativeqa.py first (the mandatory "
                f"chunking stage, mirroring MEMO's own data_processing_utils step): "
                f"python process_narrativeqa.py --narrativeqa_dir {narrativeqa_dir} --split {split}"
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
