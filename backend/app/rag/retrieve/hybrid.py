"""Dense and BM25 candidates fused by reciprocal rank.

RRF combines rankings rather than scores, so cosine similarity and BM25 weights never have
to be made commensurable — only their orderings matter.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.rag.config import HybridRetrieveConfig
from app.rag.protocols import Embedder, Retriever, VectorIndex
from app.rag.registry import register
from app.rag.retrieve.bm25 import Bm25Index, build_bm25, rank
from app.rag.retrieve.dense import apply_threshold
from app.rag.types import Chunk, Hit, RetrievalResult


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], rrf_k: int
) -> dict[str, float]:
    """Map each id to the sum of 1/(rrf_k + rank) across the rankings it appears in."""
    fused: dict[str, float] = {}
    for ranking in rankings:
        for position, identifier in enumerate(ranking):
            fused[identifier] = fused.get(identifier, 0.0) + 1.0 / (rrf_k + position + 1)
    return fused


async def retrieve_hybrid(
    query: str,
    config: HybridRetrieveConfig,
    embedder: Embedder,
    index: VectorIndex,
    bm25: Bm25Index,
    *,
    with_embeddings: bool = False,
) -> RetrievalResult:
    if not query or not query.strip():
        return RetrievalResult(query=query, hits=())

    candidates = config.fetch_k or config.top_k

    vectors = await embedder([query], "query")
    dense_hits = (
        index.search(vectors[0], candidates, with_embeddings=with_embeddings)
        if vectors
        else []
    )

    lexical = rank(bm25, query, candidates, config.bm25_k1, config.bm25_b)
    lexical_chunks: list[Chunk] = [bm25.chunks[position] for position, _ in lexical]

    by_id: dict[str, Hit] = {hit.chunk.chunk_id: hit for hit in dense_hits}
    for chunk in lexical_chunks:
        by_id.setdefault(chunk.chunk_id, Hit(chunk=chunk, score=0.0, distance=None))

    fused = reciprocal_rank_fusion(
        [
            [hit.chunk.chunk_id for hit in dense_hits],
            [chunk.chunk_id for chunk in lexical_chunks],
        ],
        config.rrf_k,
    )

    ordered = sorted(fused.items(), key=lambda pair: (-pair[1], pair[0]))
    hits = [by_id[identifier].model_copy(update={"score": score}) for identifier, score in ordered]
    hits = apply_threshold(hits, config.score_threshold)
    return RetrievalResult(query=query, hits=tuple(hits[:candidates]))


@register("retrieve", "hybrid")
def build(
    config: HybridRetrieveConfig,
    *,
    embedder: Embedder,
    index: VectorIndex,
    with_embeddings: bool = False,
) -> Retriever:
    # Built once from the index contents; rebuild the pipeline after re-indexing.
    bm25 = build_bm25(index.chunks())

    async def retriever(query: str) -> RetrievalResult:
        return await retrieve_hybrid(
            query, config, embedder, index, bm25, with_embeddings=with_embeddings
        )

    return retriever
