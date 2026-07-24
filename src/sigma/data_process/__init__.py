"""Raw dataset -> MEMO-shaped chunked corpus/questions JSONL, mirroring MEMO's own
``data_processing_utils/`` as a distinct concern from ``data_sources/``'s loaders (which
read this package's output and normalize it to ``SourceExample`` for reflection
generation/evaluation).
"""

from __future__ import annotations
