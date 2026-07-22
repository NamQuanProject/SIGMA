"""MuSiQue source adapter.

Loaded from a **local JSON/JSONL file**, matching how MeMo's own pipeline consumes it
(a local `hipporag2_dataset/musique.json` -- whose exact provenance isn't scripted in
their repo; it appears to be repackaged from the HippoRAG2 release rather than MuSiQue's
own distribution). MuSiQue isn't reliably published on Hugging Face, so this points
instead at the dataset's own official release:

Download::

    https://github.com/StonyBrookNLP/musique

See that repo's README for the actual dataset download link (a Google Drive zip at time
of writing) -- `musique_ans_v1.0_{train,dev,test}.jsonl` for answerable-only questions,
or `musique_full_v1.0_*` if you also want unanswerable ones. Point `--musique_path` at
whichever split file you download.

(I can't browse the web from here to re-verify this URL/the exact drive link still
resolve -- if it's moved, search "MuSiQue Trivedi StonyBrookNLP dataset download".)

Accepts either the official JSONL format (one JSON object per line) or a single JSON
array/object (in case you're pointed at a re-packaged copy, e.g. from HippoRAG2's
release) -- auto-detected from the file's first non-whitespace character.
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
    musique_path: str | Path,
    limit: int | None = None,
    seed: int = 42,
    **_ignored,
) -> Iterator[SourceExample]:
    path = Path(musique_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found -- --musique_path must point at a local MuSiQue JSON/JSONL "
            "file (see https://github.com/StonyBrookNLP/musique for the download link)"
        )

    rows = _read_records(path)
    if limit is not None and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)

    for row in rows:
        yield _normalize_row(row)


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    # Genuine JSONL (the official MuSiQue format) also starts with "{", since every
    # line is its own object -- so the first character can't tell JSONL apart from a
    # single JSON value spanning the whole file. Try parsing the whole file first; a
    # multi-line JSONL file fails that with "Extra data" (more than one top-level
    # value), which is exactly the signal to fall back to per-line parsing.
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        # Could be one big object keyed by id (record dicts as values) or a single record.
        if data and all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        return [data]
    raise ValueError(f"Unrecognized MuSiQue file format in {path}")


def _normalize_row(row: dict[str, Any]) -> SourceExample:
    example_id = str(row.get("id") or row.get("_id") or "")
    question = str(row.get("question") or "").strip()
    answer = str(row.get("answer") or "").strip()

    paragraphs = row.get("paragraphs") or []
    context: list[dict[str, Any]] = []
    supporting_facts: list[dict[str, Any]] = []
    num_supporting = 0
    for para in paragraphs:
        title = para.get("title", "")
        text = para.get("paragraph_text", "")
        context.append({"title": title, "sentences": [text]})
        if para.get("is_supporting"):
            num_supporting += 1
            supporting_facts.append({"title": title, "sent_id": para.get("idx")})

    return SourceExample(
        dataset=DATASET_LABEL,
        example_id=f"musique-{example_id}",
        question=question,
        answer=answer,
        context=context,
        supporting_facts=supporting_facts,
        # MuSiQue is explicitly a 2/3/4-hop benchmark; the paragraph-level is_supporting
        # count is a direct, reliable proxy for hop count when no separate field is given.
        type=f"{num_supporting}hop" if num_supporting else None,
    )
