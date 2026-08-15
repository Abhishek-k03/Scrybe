"""Index access, as the routes see it. The implementation lives in `app.rag.index`."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any

from app.rag.types import Chunk
from app.services.pipeline import get_index, reset  # noqa: F401  reset is re-exported

log = logging.getLogger("scrybe.store")


def collection() -> Any:
    """Raw Chroma collection, for the vector map's read of every stored embedding."""
    return get_index().collection


def add_chunks(chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> int:
    """Store a document's chunks, or nothing if that document is already indexed.

    Document ids are content hashes, so a repeat ingest of unchanged content would
    otherwise land a second copy and inflate every count computed over the index.
    """
    if not chunks:
        return 0

    index = get_index()
    doc_id = chunks[0].doc_id
    if index.has_doc(doc_id):
        log.info("Document %s is already indexed — nothing stored", doc_id)
        return 0
    return index.add(chunks, embeddings)


def get_all_sources() -> list[dict]:
    sources: dict[str, dict] = {}
    for chunk in get_index().chunks():
        if not chunk.doc_id:
            continue
        entry = sources.setdefault(
            chunk.doc_id,
            {
                "source_id": chunk.doc_id,
                "source_label": chunk.doc_label,
                "source_type": chunk.source_type,
                "chunk_count": 0,
            },
        )
        entry["chunk_count"] += 1
    return list(sources.values())


def delete_source(source_id: str) -> int:
    return get_index().delete_doc(source_id)


def count_total_chunks() -> int:
    return get_index().count()
