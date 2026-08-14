"""Pass-through reranker. The baseline every other reranker has to beat."""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.config import NoopRerankConfig
from app.rag.protocols import Reranker
from app.rag.registry import register
from app.rag.types import Hit


@register("rerank", "noop")
def build(config: NoopRerankConfig) -> Reranker:
    def reranker(query_embedding: Sequence[float] | None, hits: Sequence[Hit]) -> list[Hit]:
        return list(hits)

    return reranker
