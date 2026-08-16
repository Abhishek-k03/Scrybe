"""Pass-through reranker. The baseline every other reranker has to beat."""

from __future__ import annotations

from app.rag.config import NoopRerankConfig
from app.rag.protocols import Reranker
from app.rag.registry import register
from app.rag.types import Hit, RetrievalResult


@register("rerank", "noop")
def build(config: NoopRerankConfig) -> Reranker:
    async def reranker(result: RetrievalResult) -> list[Hit]:
        return list(result.hits)

    return reranker
