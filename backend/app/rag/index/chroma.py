"""Persistent ChromaDB index.

The client is built per instance rather than cached in a module global, so two differently
configured indexes can be open at once. Writes require `read_only=False`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.rag.config import ChromaIndexConfig
from app.rag.protocols import VectorIndex
from app.rag.registry import register
from app.rag.types import Chunk, Hit


class ReadOnlyIndexError(RuntimeError):
    """Raised when a write is attempted on an index opened read-only."""


def _metadata(chunk: Chunk) -> dict[str, Any]:
    return {
        "doc_id": chunk.doc_id,
        "doc_label": chunk.doc_label,
        "source_type": chunk.source_type,
        "chunk_index": chunk.chunk_index,
        "start_char": chunk.start_char,
        "end_char": chunk.end_char,
    }


def _chunk_from(document: str, meta: dict[str, Any]) -> Chunk:
    return Chunk(
        doc_id=str(meta.get("doc_id", "")),
        doc_label=str(meta.get("doc_label", "")),
        source_type=str(meta.get("source_type", "unknown")),
        chunk_index=int(meta.get("chunk_index", 0)),
        text=document or "",
        start_char=int(meta.get("start_char", 0)),
        end_char=int(meta.get("end_char", len(document or ""))),
    )


class ChromaIndex:
    def __init__(self, config: ChromaIndexConfig) -> None:
        # Imported here so `import app.rag` stays cheap and free of the chromadb stack.
        import chromadb

        self.config = config
        self._client = chromadb.PersistentClient(path=config.path)
        if config.read_only:
            self._collection = self._client.get_collection(name=config.collection)
        else:
            self._collection = self._client.get_or_create_collection(
                name=config.collection,
                metadata={"hnsw:space": config.space},
            )

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
        if self.config.read_only:
            raise ReadOnlyIndexError(
                f"index at {self.config.path!r} is read-only; set read_only=False to write"
            )
        if len(chunks) != len(embeddings):
            raise ValueError(f"{len(chunks)} chunks but {len(embeddings)} embeddings")
        if not chunks:
            return 0

        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[_metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def search(
        self, embedding: Sequence[float], top_k: int, *, with_embeddings: bool = False
    ) -> list[Hit]:
        if top_k <= 0 or self.count() == 0:
            return []

        include = ["documents", "metadatas", "distances"]
        if with_embeddings:
            include.append("embeddings")

        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=min(top_k, self.count()),
            include=include,
        )
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        vectors = (result.get("embeddings") if with_embeddings else None) or [[]]
        vectors = vectors[0] if len(vectors) else []

        hits: list[Hit] = []
        for position, (document, meta, distance) in enumerate(
            zip(documents, metadatas, distances, strict=False)
        ):
            vector = None
            if with_embeddings and position < len(vectors):
                vector = tuple(float(v) for v in vectors[position])
            hits.append(
                Hit(
                    chunk=_chunk_from(document, meta or {}),
                    score=1.0 - float(distance),
                    distance=float(distance),
                    embedding=vector,
                )
            )
        return hits

    def count(self) -> int:
        return int(self._collection.count())

    def chunks(self) -> list[Chunk]:
        result = self._collection.get(include=["documents", "metadatas"])
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            _chunk_from(document, meta or {})
            for document, meta in zip(documents, metadatas, strict=False)
        ]

    def delete_doc(self, doc_id: str) -> int:
        if self.config.read_only:
            raise ReadOnlyIndexError(
                f"index at {self.config.path!r} is read-only; set read_only=False to write"
            )
        existing = self._collection.get(where={"doc_id": doc_id}, include=[])
        removed = len(existing.get("ids", []) or [])
        if removed:
            self._collection.delete(where={"doc_id": doc_id})
        return removed


@register("index", "chroma")
def build(config: ChromaIndexConfig) -> VectorIndex:
    return ChromaIndex(config)
