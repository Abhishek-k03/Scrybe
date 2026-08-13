"""Dense vector retrieval.

Returns `fetch_k` candidates when set, otherwise `top_k`. Trimming to `top_k` happens after
reranking, so a reranker has more than the final answer to choose from.
"""

from __future__ import annotations

from app.rag.config import DenseRetrieveConfig
from app.rag.protocols import Embedder, Retriever, VectorIndex
from app.rag.registry import register
from app.rag.types import Hit, RetrievalResult


def apply_threshold(hits: list[Hit], threshold: float | None) -> list[Hit]:
    """Drop hits below the score floor, letting retrieval return nothing."""
    if threshold is None:
        return hits
    return [hit for hit in hits if hit.score >= threshold]


async def retrieve_dense(
    query: str,
    config: DenseRetrieveConfig,
    embedder: Embedder,
    index: VectorIndex,
    *,
    with_embeddings: bool = False,
) -> RetrievalResult:
    if not query or not query.strip():
        return RetrievalResult(query=query, hits=())

    vectors = await embedder([query], "query")
    if not vectors:
        return RetrievalResult(query=query, hits=())

    candidates = config.fetch_k or config.top_k
    hits = index.search(vectors[0], candidates, with_embeddings=with_embeddings)
    return RetrievalResult(query=query, hits=tuple(apply_threshold(hits, config.score_threshold)))


@register("retrieve", "dense")
def build(
    config: DenseRetrieveConfig,
    *,
    embedder: Embedder,
    index: VectorIndex,
    with_embeddings: bool = False,
) -> Retriever:
    async def retriever(query: str) -> RetrievalResult:
        return await retrieve_dense(
            query, config, embedder, index, with_embeddings=with_embeddings
        )

    return retriever
