"""Jina embeddings over HTTP.

Retries on rate limits and server errors; a failed batch mid-run would otherwise leave an
index half-built with no indication which chunks are missing.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

import httpx

from app.rag.cache import DiskCache
from app.rag.config import JinaEmbedConfig
from app.rag.embed.cached import cached_embedder
from app.rag.protocols import Embedder, EmbedKind
from app.rag.registry import register

API_URL = "https://api.jina.ai/v1/embeddings"
RETRY_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


def task_for(config: JinaEmbedConfig, kind: EmbedKind) -> str:
    return config.query_task if kind == "query" else config.passage_task


async def _post_batch(
    client: httpx.AsyncClient,
    config: JinaEmbedConfig,
    api_key: str,
    batch: Sequence[str],
    task: str,
) -> list[list[float]]:
    payload = {"model": config.model, "task": task, "input": list(batch)}
    headers = {"Authorization": f"Bearer {api_key}"}

    for attempt in range(config.max_retries + 1):
        response = await client.post(API_URL, headers=headers, json=payload)
        if response.status_code in RETRY_STATUS and attempt < config.max_retries:
            await asyncio.sleep(2.0**attempt)
            continue
        response.raise_for_status()
        data = response.json()["data"]
        # The API echoes an index per item; sort on it rather than trusting arrival order.
        ordered = sorted(data, key=lambda item: item.get("index", 0))
        return [item["embedding"] for item in ordered]

    raise RuntimeError("unreachable: retry loop exited without returning")


async def embed_texts(
    texts: Sequence[str],
    config: JinaEmbedConfig,
    api_key: str,
    kind: EmbedKind = "passage",
    *,
    timeout: float = 60.0,
) -> list[list[float]]:
    if not texts:
        return []
    if not api_key:
        raise RuntimeError("JINA_API_KEY is not configured")

    task = task_for(config, kind)
    out: list[list[float]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for start in range(0, len(texts), config.batch_size):
            batch = texts[start : start + config.batch_size]
            out.extend(await _post_batch(client, config, api_key, batch, task))

    if len(out) != len(texts):
        raise RuntimeError(f"expected {len(texts)} embeddings, got {len(out)}")
    return out


@register("embed", "jina")
def build(config: JinaEmbedConfig, api_key: str = "") -> Embedder:
    async def embedder(texts: Sequence[str], kind: EmbedKind = "passage") -> list[list[float]]:
        return await embed_texts(texts, config, api_key, kind)

    if config.cache_dir is None:
        return embedder
    return cached_embedder(
        embedder,
        DiskCache(config.cache_dir),
        model=config.model,
        task_for=lambda kind: task_for(config, kind),
    )
