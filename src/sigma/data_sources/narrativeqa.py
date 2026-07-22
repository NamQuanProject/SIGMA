"""NarrativeQA source adapter.

Loaded from a **local checkout** of the official NarrativeQA GitHub repo, matching how
MeMo's own pipeline consumes it (a cloned repo + `documents.csv`/`qaps.csv`, not a
Hugging Face dataset -- NarrativeQA isn't reliably published there).

Download::

    git clone https://github.com/google-deepmind/narrativeqa

That ships `documents.csv`, `qaps.csv`, and `third_party/wikipedia/summaries.csv`
directly in the repo -- no extra download step needed for those three files. We use
each story's Wikipedia plot **summary** as context, not the full book/script text, so
you do NOT need to run the repo's separate `download_stories.sh` (which fetches full
story text from Project Gutenberg / script archives -- slow, and not needed here).

(I can't browse the web from here to re-verify this URL still resolves -- if the repo
has moved, `--narrativeqa_dir` just needs to point at wherever you actually cloned it;
the three CSV filenames above are what this loader looks for.)
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterator

from .base import SourceExample

DATASET_LABEL = "narrativeqa"

# NarrativeQA's own documents.csv/qaps.csv "set" column uses "valid", not "validation".
_SPLIT_ALIASES = {"validation": "valid", "val": "valid", "dev": "valid"}


def load_examples(
    *,
    narrativeqa_dir: str | Path,
    split: str = "train",
    limit: int | None = None,
    seed: int = 42,
    **_ignored,
) -> Iterator[SourceExample]:
    root = Path(narrativeqa_dir)
    documents_csv = root / "documents.csv"
    qaps_csv = root / "qaps.csv"
    summaries_csv = root / "third_party" / "wikipedia" / "summaries.csv"
    for path in (documents_csv, qaps_csv, summaries_csv):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path} -- --narrativeqa_dir must point at a checkout of "
                "https://github.com/google-deepmind/narrativeqa (git clone it; no further "
                "download step is needed for these three CSVs specifically)"
            )

    want_set = _SPLIT_ALIASES.get(split, split)

    summary_by_doc: dict[str, str] = {}
    with summaries_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            summary_by_doc[row["document_id"]] = row.get("summary", "")

    doc_ids_in_split: set[str] = set()
    with documents_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("set") == want_set:
                doc_ids_in_split.add(row["document_id"])

    if not doc_ids_in_split:
        raise ValueError(
            f"No documents found with set={want_set!r} in {documents_csv} -- check "
            f"--split (NarrativeQA's own splits are 'train'/'valid'/'test')"
        )

    rows = []
    with qaps_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("document_id") in doc_ids_in_split:
                rows.append(row)

    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)

    for index, row in enumerate(rows):
        doc_id = row["document_id"]
        summary_text = summary_by_doc.get(doc_id, "").strip()
        question = (row.get("question") or "").strip()
        # NarrativeQA gives two independently-written reference answers; use the first
        # as the primary gold answer (same convention as our other single-answer sources).
        answer = (row.get("answer1") or "").strip()

        yield SourceExample(
            dataset=DATASET_LABEL,
            example_id=f"narrativeqa-{doc_id}-{index}",
            question=question,
            answer=answer,
            context=[{"title": doc_id, "sentences": [summary_text]}] if summary_text else [],
        )
