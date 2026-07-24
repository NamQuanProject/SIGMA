"""Word-count-based overlapping text chunking, ported verbatim from MEMO's
``chunk_text`` (identical function, duplicated in both
``MeMo/data_processing_utils/convert_narrativeqa_to_chunks_jsonl.py`` and
``convert_musique_to_chunks_jsonl.py``). Shared here since both of our processing
scripts (``process_narrativeqa.py``, ``process_musique.py``) need the same algorithm.
"""

from __future__ import annotations


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by word count.

    With MEMO's own default ``chunk_size=6400`` words, this is a no-op (returns
    ``[text]`` unchanged) for anything shorter than that -- which in practice is nearly
    everything we feed it (NarrativeQA summaries, MuSiQue paragraphs), since neither
    comes close to 6400 words. It's still the right thing to run: it's MEMO's actual
    processing step, and it does its job (splitting) correctly on the rare input that's
    actually long enough to need it.
    """

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        chunks.append(" ".join(chunk_words))
        if start + chunk_size >= len(words):
            break
    return chunks
