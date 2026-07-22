"""Process raw MuSiQue into MEMO-shaped chunked corpus/questions JSONL -- the mandatory
first stage before reflection generation, mirroring MEMO's own
``data_processing_utils/convert_musique_to_chunks_jsonl.py``.

MuSiQue's paragraphs are already short (well under the default 6400-word chunk size), so
chunking is almost always a no-op (one chunk per paragraph) -- expected, not a bug; it's
still MEMO's real algorithm, faithfully applied.

Output (same schema as MEMO's own script, minus their hardcoded ``_1000`` filename
suffix -- a leftover from one specific run size in their repo that doesn't mean anything
generically, so we don't perpetuate it):

- ``musique_corpus_chunks.jsonl``, one line per chunk:
  ``{"docid", "text", "url"}``
- ``musique_questions_chunks.jsonl``, one line per question:
  ``{"query_id", "question", "answers", "document_id", "hop", "evidence_docs",
  "gold_docs", "negative_docs"}``
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from loguru import logger

from ..utils.logging_setup import setup_logging
from .chunking import chunk_text


def _get_hop_count(qid: str) -> str:
    m = re.match(r"(\d+hop)", qid)
    return m.group(1) if m else "unknown"


def _read_records(path: Path) -> list[dict[str, Any]]:
    """Accepts either the official one-JSON-object-per-line format or a single JSON
    array/object (e.g. a re-packaged copy) -- same tolerant reading MEMO's own script
    doesn't need (it assumes a plain JSON array), but our downloaded copy might be either.
    """

    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if data and all(isinstance(v, dict) for v in data.values()):
            return list(data.values())
        return [data]
    raise ValueError(f"Unrecognized MuSiQue file format in {path}")


def process(
    *,
    musique_path: Path,
    output_dir: Path,
    chunk_size: int = 6400,
    overlap: int = 640,
) -> tuple[Path, Path]:
    if not musique_path.is_file():
        raise FileNotFoundError(
            f"{musique_path} not found -- --musique_path must point at a local MuSiQue "
            "JSON/JSONL file (see https://github.com/StonyBrookNLP/musique for the download link)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = output_dir / "musique_corpus_chunks.jsonl"
    questions_path = output_dir / "musique_questions_chunks.jsonl"

    examples = _read_records(musique_path)
    logger.info(f"Loaded {len(examples)} MuSiQue examples from {musique_path}")
    logger.info(f"Chunking (chunk_size={chunk_size} words, overlap={overlap} words)...")

    total_chunks = total_supporting_chunks = total_negative_chunks = 0

    with corpus_path.open("w", encoding="utf-8") as corpus_f, questions_path.open("w", encoding="utf-8") as questions_f:
        for entry in examples:
            qid = str(entry.get("id") or entry.get("_id") or "")
            hop = _get_hop_count(qid)
            paragraphs = entry.get("paragraphs") or []

            para_chunk_map: dict[str, list[str]] = {}
            for para in paragraphs:
                para_base = f"{qid}_para{para.get('idx')}"
                chunks = chunk_text(para.get("paragraph_text", ""), chunk_size, overlap)
                chunk_docids = []
                for i, chunk in enumerate(chunks):
                    chunk_docid = f"{para_base}_chunk{i}"
                    chunk_docids.append(chunk_docid)
                    corpus_f.write(
                        json.dumps({"docid": chunk_docid, "text": chunk, "url": para.get("title", "")}, ensure_ascii=False)
                        + "\n"
                    )
                para_chunk_map[para_base] = chunk_docids
                total_chunks += len(chunks)
                if para.get("is_supporting"):
                    total_supporting_chunks += len(chunks)
                else:
                    total_negative_chunks += len(chunks)

            evidence_docs, negative_docs = [], []
            for para in paragraphs:
                para_base = f"{qid}_para{para.get('idx')}"
                chunk_refs = [{"docid": cid} for cid in para_chunk_map[para_base]]
                (evidence_docs if para.get("is_supporting") else negative_docs).extend(chunk_refs)

            answers = [entry.get("answer", "")] + list(entry.get("answer_aliases") or [])
            questions_f.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "question": entry.get("question", ""),
                        "answers": answers,
                        "document_id": qid,
                        "hop": hop,
                        "evidence_docs": evidence_docs,
                        "gold_docs": evidence_docs,
                        "negative_docs": negative_docs,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    logger.info(
        f"Total chunks: {total_chunks} (supporting: {total_supporting_chunks}, negative: {total_negative_chunks})"
    )
    logger.info(f"Corpus written to {corpus_path}")
    logger.info(f"Questions written to {questions_path}")
    return corpus_path, questions_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a raw MuSiQue JSON/JSONL file into MEMO-shaped chunked JSONL."
    )
    parser.add_argument("--musique_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, default=Path("data/MuSiQue"))
    parser.add_argument("--chunk_size", type=int, default=6400, help="Chunk size in words (MEMO default: 6400)")
    parser.add_argument("--overlap", type=int, default=640, help="Overlap between chunks in words (MEMO default: 640)")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging("process_musique", log_dir=args.log_dir)
    process(
        musique_path=args.musique_path,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )


if __name__ == "__main__":
    main()
