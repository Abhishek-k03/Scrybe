"""Embedding, as the routes see it. The implementation lives in `app.rag.embed`."""

from __future__ import annotations

import logging

from app.rag.protocols import EmbedKind
from app.services import pipeline

log = logging.getLogger("scrybe.embedder")


async def embed_texts(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    if not texts:
        return []
    kind: EmbedKind = "query" if task.endswith(".query") else "passage"
    embeddings = await pipeline.get().embedder(texts, kind)
    log.info("Embedded %d texts (task=%s)", len(texts), task)
    return embeddings
