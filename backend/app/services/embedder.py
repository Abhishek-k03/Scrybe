import logging

import httpx

from app.core.config import settings

log = logging.getLogger("weboracle.embedder")

JINA_URL = "https://api.jina.ai/v1/embeddings"
MODEL = "jina-embeddings-v3"
BATCH = 32


async def embed_texts(texts: list[str], task: str = "retrieval.passage") -> list[list[float]]:
    if not texts:
        return []
    if not settings.JINA_API_KEY:
        raise RuntimeError("JINA_API_KEY is not configured")

    embeddings: list[list[float]] = []
    headers = {"Authorization": f"Bearer {settings.JINA_API_KEY}"}

    async with httpx.AsyncClient(timeout=60) as client:
        for i in range(0, len(texts), BATCH):
            batch = texts[i : i + BATCH]
            payload = {"model": MODEL, "task": task, "input": batch}
            resp = await client.post(JINA_URL, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            embeddings.extend([d["embedding"] for d in data["data"]])

    log.info("Embedded %d texts (task=%s)", len(texts), task)
    return embeddings
