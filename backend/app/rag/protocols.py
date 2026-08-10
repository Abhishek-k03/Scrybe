"""Callable shapes each stage implements.

Stages are plain functions closed over their config by a factory, not classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from app.rag.types import Chunk, Document

EmbedKind = Literal["passage", "query"]


class Chunker(Protocol):
    def __call__(self, doc: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    async def __call__(
        self, texts: Sequence[str], kind: EmbedKind = "passage"
    ) -> list[list[float]]: ...
