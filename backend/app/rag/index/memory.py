"""Exact in-process cosine search.

No files, no approximate nearest neighbours, so results are reproducible run to run. This is
the reference the Chroma backend is checked against.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.rag.config import MemoryIndexConfig
from app.rag.protocols import VectorIndex
from app.rag.registry import register
from app.rag.types import Chunk, Hit


def _normalise(block: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(block, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return block / norms


class MemoryIndex:
    def __init__(self, config: MemoryIndexConfig) -> None:
        self.config = config
        self._chunks: list[Chunk] = []
        self._vectors: np.ndarray | None = None

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return 0

        block = np.asarray(embeddings, dtype=np.float64)
        if block.ndim != 2:
            raise ValueError("embeddings must be a 2-d array")
        if self._vectors is not None and block.shape[1] != self._vectors.shape[1]:
            raise ValueError(
                f"embedding width {block.shape[1]} does not match index width "
                f"{self._vectors.shape[1]}"
            )

        block = _normalise(block)
        self._vectors = block if self._vectors is None else np.vstack([self._vectors, block])
        self._chunks.extend(chunks)
        return len(chunks)

    def search(self, embedding: Sequence[float], top_k: int) -> list[Hit]:
        if self._vectors is None or not self._chunks or top_k <= 0:
            return []

        query = np.asarray(embedding, dtype=np.float64)
        if query.shape[0] != self._vectors.shape[1]:
            raise ValueError(
                f"query width {query.shape[0]} does not match index width "
                f"{self._vectors.shape[1]}"
            )
        norm = float(np.linalg.norm(query)) or 1.0
        similarities = self._vectors @ (query / norm)

        # Stable sort so equal scores keep insertion order rather than varying per run.
        order = np.argsort(-similarities, kind="stable")[: min(top_k, len(self._chunks))]
        return [
            Hit(
                chunk=self._chunks[i],
                score=float(similarities[i]),
                distance=float(1.0 - similarities[i]),
            )
            for i in order
        ]

    def count(self) -> int:
        return len(self._chunks)

    def delete_doc(self, doc_id: str) -> int:
        keep = [i for i, chunk in enumerate(self._chunks) if chunk.doc_id != doc_id]
        removed = len(self._chunks) - len(keep)
        if not removed:
            return 0
        self._chunks = [self._chunks[i] for i in keep]
        self._vectors = self._vectors[keep] if keep else None
        return removed


@register("index", "memory")
def build(config: MemoryIndexConfig) -> VectorIndex:
    return MemoryIndex(config)
