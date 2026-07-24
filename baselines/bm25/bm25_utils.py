"""Minimal, dependency-free BM25 (Robertson/Sparck-Jones) ranking.

Scoped to ranking a *handful* of candidate context blocks per question (SIGMA's
data_sources already narrow each example down to its own context, not a corpus-wide
index -- see run_bm25_baseline.py), so a full inverted-index library like MEMO's own
Pyserini-based retrieval is unnecessary overhead here; plain stdlib is enough.
"""

from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["bm25_rank"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def bm25_rank(query: str, documents: list[str], *, k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Return ``documents`` indices sorted by BM25 score against ``query``, best first."""

    tokenized_docs = [_tokenize(doc) for doc in documents]
    n_docs = len(tokenized_docs)
    if n_docs == 0:
        return []

    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avg_len = sum(doc_lengths) / n_docs

    doc_freq: Counter[str] = Counter()
    for tokens in tokenized_docs:
        doc_freq.update(set(tokens))

    scores = [0.0] * n_docs
    for term in _tokenize(query):
        df = doc_freq.get(term, 0)
        if df == 0:
            continue
        idf = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        for i, tokens in enumerate(tokenized_docs):
            tf = tokens.count(term)
            if tf == 0:
                continue
            length_norm = 1 - b + b * (doc_lengths[i] / avg_len if avg_len else 0.0)
            scores[i] += idf * (tf * (k1 + 1)) / (tf + k1 * length_norm)

    return sorted(range(n_docs), key=lambda i: scores[i], reverse=True)
