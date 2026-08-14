"""Maximal marginal relevance.

Trades relevance against redundancy, which matters here because overlapping chunks make
near-duplicate neighbours common: the top hits are often the same passage shifted by the
overlap window.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.rag.config import MmrRerankConfig
from app.rag.protocols import Reranker
from app.rag.registry import register
from app.rag.types import Hit


class MissingEmbeddingError(ValueError):
    """Raised when a candidate arrives without the vector MMR needs."""


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(hits: Sequence[Hit], config: MmrRerankConfig) -> list[Hit]:
    if not hits:
        return []

    missing = [hit.chunk.chunk_id for hit in hits if hit.embedding is None]
    if missing:
        raise MissingEmbeddingError(
            f"MMR needs embeddings; {len(missing)} candidate(s) arrived without one "
            f"(first: {missing[0]}). Retrieve with with_embeddings=True."
        )

    remaining = list(hits)
    selected: list[Hit] = []
    limit = min(config.top_k, len(remaining))

    while remaining and len(selected) < limit:
        best_index = 0
        best_value = -math.inf
        for position, candidate in enumerate(remaining):
            redundancy = (
                max(cosine(candidate.embedding, chosen.embedding) for chosen in selected)
                if selected
                else 0.0
            )
            value = config.lambda_mult * candidate.score - (1.0 - config.lambda_mult) * redundancy
            # Strict > keeps the earlier candidate on ties, so order stays deterministic.
            if value > best_value:
                best_value = value
                best_index = position
        selected.append(remaining.pop(best_index))

    return selected


@register("rerank", "mmr")
def build(config: MmrRerankConfig) -> Reranker:
    def reranker(query_embedding: Sequence[float] | None, hits: Sequence[Hit]) -> list[Hit]:
        return rerank_mmr(hits, config)

    return reranker
