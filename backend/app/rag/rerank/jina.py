"""Jina cross-encoder reranking over HTTP.

The retriever ranks by distance in a fixed embedding space; this scores each candidate
against the query directly, which is what lets it separate passages the bi-encoder put
side by side. Only worth running over a candidate pool wider than the final `top_k`, so
`fetch_k` on the retrieve config is what gives this stage something to do.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from app.rag.cache import DiskCache, content_key
from app.rag.config import JinaRerankConfig
from app.rag.protocols import Reranker
from app.rag.registry import register
from app.rag.types import Hit, RetrievalResult

API_URL = "https://api.jina.ai/v1/rerank"
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class RerankResponseError(RuntimeError):
    """Raised when the API returns a set of scores that does not match what was sent."""


def cache_key(model: str, query: str, documents: Sequence[str]) -> str:
    """Key on the candidate texts, not their ids.

    A chunk id is `doc_id-index`, which repeats across chunk sizes over different text.
    Keying on ids would serve a 400-char sweep the scores computed for the 1600-char one.
    """
    return content_key("jina_rerank", model, query, *documents)


def scores_from_response(payload: dict, expected: int) -> list[float]:
    """Relevance scores in the order the documents were sent."""
    results = payload.get("results", [])
    if len(results) != expected:
        raise RerankResponseError(f"expected {expected} scores, got {len(results)}")

    scores = [0.0] * expected
    seen = set()
    for item in results:
        index = item["index"]
        if not 0 <= index < expected or index in seen:
            raise RerankResponseError(f"response index {index} is out of range or repeated")
        seen.add(index)
        scores[index] = float(item["relevance_score"])
    return scores


async def _post(
    client: httpx.AsyncClient,
    config: JinaRerankConfig,
    api_key: str,
    query: str,
    documents: Sequence[str],
) -> list[float]:
    payload = {
        "model": config.model,
        "query": query,
        "documents": list(documents),
        # We hold the candidate texts already; echoing them back only costs bandwidth.
        "return_documents": False,
    }
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(config.max_retries + 1):
        response = await client.post(API_URL, headers=headers, json=payload)
        if response.status_code in RETRY_STATUS and attempt < config.max_retries:
            await asyncio.sleep(2.0**attempt)
            continue
        response.raise_for_status()
        return scores_from_response(response.json(), len(documents))

    raise RuntimeError("unreachable: retry loop exited without returning")


async def score_documents(
    query: str,
    documents: Sequence[str],
    config: JinaRerankConfig,
    api_key: str,
    *,
    cache: DiskCache | None = None,
    timeout: float = 60.0,
) -> list[float]:
    """Relevance scores aligned to `documents`, from the cache when it has them.

    Every candidate is scored and cached, and `top_k` is applied afterwards, so narrowing
    the final cut is free and widening it only costs whatever the pool grew by.
    """
    if not documents:
        return []
    if not query.strip():
        return [0.0] * len(documents)

    key = cache_key(config.model, query, documents) if cache is not None else ""
    if cache is not None:
        cached = cache.get(key)
        # A stored entry of the wrong length is from a different candidate set; a hash
        # collision, in other words. Refetch rather than mis-score.
        if isinstance(cached, list) and len(cached) == len(documents):
            return [float(value) for value in cached]

    if not api_key:
        raise RuntimeError("JINA_API_KEY is not configured")

    async with httpx.AsyncClient(timeout=timeout) as client:
        scores = await _post(client, config, api_key, query, documents)

    if cache is not None:
        cache.set(key, scores)
    return scores


def order_by_score(hits: Sequence[Hit], scores: Sequence[float]) -> list[Hit]:
    """Sort descending by relevance, replacing `Hit.score` and leaving `distance` alone.

    `distance` stays as the retriever measured it: the two numbers come from different
    models and overwriting one with the other would make the artifact unreadable.
    """
    rescored = [
        hit.model_copy(update={"score": score}) for hit, score in zip(hits, scores, strict=True)
    ]
    # Negated key rather than reverse=True: Python's sort is stable, so this keeps the
    # retriever's order among ties instead of flipping it.
    return sorted(rescored, key=lambda hit: -hit.score)


@register("rerank", "jina_rerank")
def build(config: JinaRerankConfig, api_key: str = "") -> Reranker:
    cache = DiskCache(config.cache_dir) if config.cache_dir is not None else None

    async def reranker(result: RetrievalResult) -> list[Hit]:
        if not result.hits:
            return []
        documents = [hit.chunk.text for hit in result.hits]
        scores = await score_documents(result.query, documents, config, api_key, cache=cache)
        ordered = order_by_score(result.hits, scores)
        if config.score_threshold is not None:
            ordered = [hit for hit in ordered if hit.score >= config.score_threshold]
        return ordered[: config.top_k]

    return reranker
