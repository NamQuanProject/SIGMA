"""Process raw NarrativeQA into MEMO-shaped chunked corpus/questions JSONL -- the
mandatory first stage before reflection generation, mirroring MEMO's own
``data_processing_utils/convert_narrativeqa_to_chunks_jsonl.py``.

Adapted for what we actually have: MEMO's script reads full scraped story text from a
``tmp/<doc_id>.content`` directory (produced by a *separate* scraping step in the
original `deepmind/narrativeqa` repo, not provided by MEMO itself, and not something we
fetch -- see the top-level README). We chunk each story's Wikipedia plot **summary**
instead (``third_party/wikipedia/summaries.csv``). Since summaries are far shorter than
the default 6400-word chunk size, chunking will almost always be a no-op (one chunk per
document) -- that's expected, not a bug; it's still MEMO's real algorithm, faithfully
applied to shorter input.

Output (identical schema to MEMO's own script):

- ``narrativeqa_<split>_corpus_chunks.jsonl``, one line per chunk:
  ``{"docid", "text", "url"}``
- ``narrativeqa_<split>_questions_chunks.jsonl``, one line per QA pair:
  ``{"query_id", "question", "answers", "document_id", "evidence_docs", "gold_docs"}``
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from loguru import logger

from ..utils.logging_setup import setup_logging
from .chunking import chunk_text

# NarrativeQA's own "set" column uses "valid", not "validation".
_SPLIT_ALIASES = {"validation": "valid", "val": "valid", "dev": "valid"}


def process(
    *,
    narrativeqa_dir: Path,
    output_dir: Path,
    split: str = "train",
    chunk_size: int = 6400,
    overlap: int = 640,
) -> tuple[Path, Path]:
    documents_csv = narrativeqa_dir / "documents.csv"
    qaps_csv = narrativeqa_dir / "qaps.csv"
    summaries_csv = narrativeqa_dir / "third_party" / "wikipedia" / "summaries.csv"
    for path in (documents_csv, qaps_csv, summaries_csv):
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing {path} -- --narrativeqa_dir must point at a checkout of "
                "https://github.com/google-deepmind/narrativeqa (git clone it; no further "
                "download step is needed for these three CSVs specifically)"
            )

    want_set = _SPLIT_ALIASES.get(split, split)
    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / f"narrativeqa_{split}_corpus_chunks.jsonl"
    questions_path = output_dir / f"narrativeqa_{split}_questions_chunks.jsonl"

    summary_by_doc: dict[str, str] = {}
    with summaries_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            summary_by_doc[row["document_id"]] = row.get("summary", "")

    doc_ids_in_split: set[str] = set()
    with documents_csv.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if split == "all" or row.get("set") == want_set:
                doc_ids_in_split.add(row["document_id"])

    if not doc_ids_in_split:
        raise ValueError(f"No documents found with set={want_set!r} in {documents_csv} -- check --split")

    logger.info(f"Chunking (split={split!r}, chunk_size={chunk_size}, overlap={overlap})...")

    doc_chunk_map: dict[str, list[str]] = {}
    total_chunks = 0
    skipped_docs = 0

    with corpus_path.open("w", encoding="utf-8") as out_f:
        for doc_id in sorted(doc_ids_in_split):
            summary = summary_by_doc.get(doc_id, "").strip()
            if not summary:
                skipped_docs += 1
                continue
            chunks = chunk_text(summary, chunk_size, overlap)
            chunk_docids = []
            for i, chunk in enumerate(chunks):
                chunk_docid = f"{doc_id}_chunk{i}"
                chunk_docids.append(chunk_docid)
                out_f.write(json.dumps({"docid": chunk_docid, "text": chunk, "url": doc_id}, ensure_ascii=False) + "\n")
            doc_chunk_map[doc_id] = chunk_docids
            total_chunks += len(chunks)

    logger.info(
        f"Documents loaded: {len(doc_chunk_map)}, skipped: {skipped_docs}, "
        f"total chunks: {total_chunks}, avg chunks/doc: {total_chunks / max(len(doc_chunk_map), 1):.2f}"
    )
    logger.info(f"Corpus written to {corpus_path}")

    written, skipped_qns = 0, 0
    with qaps_csv.open(newline="", encoding="utf-8") as qf, questions_path.open("w", encoding="utf-8") as out_f:
        for i, row in enumerate(csv.DictReader(qf)):
            doc_id = row["document_id"]
            if split != "all" and row.get("set") != want_set:
                skipped_qns += 1
                continue
            if doc_id not in doc_chunk_map:
                skipped_qns += 1
                continue
            chunk_refs = [{"docid": cid} for cid in doc_chunk_map[doc_id]]
            out_f.write(
                json.dumps(
                    {
                        "query_id": f"narrativeqa_{doc_id}_q{i}",
                        "question": row["question"],
                        "answers": [row.get("answer1", ""), row.get("answer2", "")],
                        "document_id": doc_id,
                        "evidence_docs": chunk_refs,
                        "gold_docs": chunk_refs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    logger.info(f"Questions written: {written}, skipped: {skipped_qns} -- {questions_path}")
    return corpus_path, questions_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert raw NarrativeQA (documents.csv/qaps.csv/summaries.csv) into MEMO-shaped chunked JSONL."
    )
    parser.add_argument("--narrativeqa_dir", type=Path, default=Path("data/NarrativeQA"))
    parser.add_argument(
        "--output_dir", type=Path, default=None, help="Defaults to --narrativeqa_dir (write alongside the raw data)"
    )
    parser.add_argument("--split", default="train", choices=["train", "test", "valid", "validation", "all"])
    parser.add_argument("--chunk_size", type=int, default=6400, help="Chunk size in words (MEMO default: 6400)")
    parser.add_argument("--overlap", type=int, default=640, help="Overlap between chunks in words (MEMO default: 640)")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("process_narrativeqa", log_dir=args.log_dir)
    output_dir = args.output_dir or args.narrativeqa_dir
    process(
        narrativeqa_dir=args.narrativeqa_dir,
        output_dir=output_dir,
        split=args.split,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
