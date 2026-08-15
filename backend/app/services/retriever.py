"""Retrieval, as the routes see it. The implementation lives in `app.rag.retrieve`."""

from __future__ import annotations

import logging

from app.rag.types import Hit
from app.services import pipeline

log = logging.getLogger("scrybe.retriever")


def as_dict(hit: Hit) -> dict:
    """Flatten a Hit into the shape the routes and the frontend already speak."""
    return {
        "chunk_id": hit.chunk.chunk_id,
        "text": hit.chunk.text,
        "source_id": hit.chunk.doc_id,
        "source_label": hit.chunk.doc_label,
        "source_type": hit.chunk.source_type,
        "chunk_index": hit.chunk.chunk_index,
        # Cosine distance, lower is better. `score` is the same value as similarity.
        "distance": hit.distance if hit.distance is not None else 1.0 - hit.score,
        "score": hit.score,
    }


async def retrieve(question: str, top_k: int = 5) -> list[dict]:
    result = await pipeline.get().retrieve(question, top_k=top_k)
    log.info("Retrieved %d chunks for query: %s", len(result), question[:80])
    return [as_dict(hit) for hit in result.hits]
