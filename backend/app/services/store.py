import logging
from threading import Lock

import chromadb

from app.core.config import settings

log = logging.getLogger("weboracle.store")

_client = None
_collection = None
_lock = Lock()


def _get_collection():
    global _client, _collection
    if _collection is None:
        with _lock:
            if _collection is None:
                _client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
                _collection = _client.get_or_create_collection(
                    name=settings.COLLECTION_NAME,
                    metadata={"hnsw:space": "cosine"},
                )
                log.info("ChromaDB ready at %s (collection=%s)", settings.CHROMA_PATH, settings.COLLECTION_NAME)
    return _collection


def add_chunks(
    source_id: str,
    source_label: str,
    source_type: str,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> int:
    if not chunks:
        return 0
    coll = _get_collection()
    ids = [f"{source_id}-{c['chunk_index']}" for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "source_id": source_id,
            "source_label": source_label,
            "source_type": source_type,
            "chunk_index": c["chunk_index"],
        }
        for c in chunks
    ]
    coll.add(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)
    return len(chunks)


def get_all_sources() -> list[dict]:
    coll = _get_collection()
    result = coll.get(include=["metadatas"])
    metas = result.get("metadatas") or []
    sources: dict[str, dict] = {}
    for m in metas:
        sid = m.get("source_id")
        if not sid:
            continue
        if sid not in sources:
            sources[sid] = {
                "source_id": sid,
                "source_label": m.get("source_label", ""),
                "source_type": m.get("source_type", "unknown"),
                "chunk_count": 0,
            }
        sources[sid]["chunk_count"] += 1
    return list(sources.values())


def delete_source(source_id: str) -> int:
    coll = _get_collection()
    existing = coll.get(where={"source_id": source_id}, include=[])
    n = len(existing.get("ids", []) or [])
    if n:
        coll.delete(where={"source_id": source_id})
    return n


def count_total_chunks() -> int:
    return _get_collection().count()


def query_similar(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    coll = _get_collection()
    result = coll.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    dists = (result.get("distances") or [[]])[0]

    hits: list[dict] = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(
            {
                "text": doc,
                "source_id": meta.get("source_id", ""),
                "source_label": meta.get("source_label", ""),
                "source_type": meta.get("source_type", "unknown"),
                "chunk_index": meta.get("chunk_index", 0),
                "distance": float(dist),
            }
        )
    return hits
