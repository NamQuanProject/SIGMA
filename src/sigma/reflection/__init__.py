"""Reflection generation: turns raw documents into the QA pairs (Q_final) that
bootstrap-and-consolidate trains on. ``dataset.py`` is the shared loader every later
stage reads; the rest of this package is only used by ``reflections.py`` (the stage-4
CLI, one level up) to produce that data in the first place.
"""

from __future__ import annotations
