"""Callable shapes each stage implements.

Stages are plain functions closed over their config by a factory, not classes.
"""

from __future__ import annotations

from typing import Protocol

from app.rag.types import Chunk, Document


class Chunker(Protocol):
    def __call__(self, doc: Document) -> list[Chunk]: ...
