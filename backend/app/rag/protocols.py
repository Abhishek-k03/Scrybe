"""Callable shapes each stage implements.

Stages are plain functions closed over their config by a factory, not classes.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol

from app.rag.types import Chunk, Document, Hit, RetrievalResult

EmbedKind = Literal["passage", "query"]


class Chunker(Protocol):
    def __call__(self, doc: Document) -> list[Chunk]: ...


class Embedder(Protocol):
    async def __call__(
        self, texts: Sequence[str], kind: EmbedKind = "passage"
    ) -> list[list[float]]: ...


class VectorIndex(Protocol):
    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int: ...
    def search(
        self, embedding: Sequence[float], top_k: int, *, with_embeddings: bool = False
    ) -> list[Hit]: ...
    def count(self) -> int: ...
    def delete_doc(self, doc_id: str) -> int: ...
    def chunks(self) -> list[Chunk]: ...
    def has_doc(self, doc_id: str) -> bool: ...


class Retriever(Protocol):
    async def __call__(self, query: str) -> RetrievalResult: ...


class Reranker(Protocol):
    """Reorders candidates.

    Takes the whole `RetrievalResult` rather than a subset of it: a cross-encoder scores the
    query text against each candidate, MMR only compares candidates to each other, and a
    future stage may want something else again. Passing everything keeps the signature still.
    """

    async def __call__(self, result: RetrievalResult) -> list[Hit]: ...
