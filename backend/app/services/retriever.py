import logging

from app.services.embedder import embed_texts
from app.services.store import query_similar

log = logging.getLogger("weboracle.retriever")


async def retrieve(question: str, top_k: int = 5) -> list[dict]:
    if not question or not question.strip():
        return []

    embeddings = await embed_texts([question], task="retrieval.query")
    if not embeddings:
        return []

    hits = query_similar(embeddings[0], top_k=top_k)
    log.info("Retrieved %d chunks for query: %s", len(hits), question[:80])
    return hits
