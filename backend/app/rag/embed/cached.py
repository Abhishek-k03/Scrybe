"""Disk-cache wrapper around any embedder.

Keys include the model and the task string, so passage and query embeddings of the same
text never collide, and changing the model invalidates everything automatically.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from app.rag.cache import DiskCache, content_key
from app.rag.protocols import Embedder, EmbedKind


def cached_embedder(
    inner: Embedder,
    cache: DiskCache,
    *,
    model: str,
    task_for: Callable[[EmbedKind], str],
) -> Embedder:
    async def embedder(texts: Sequence[str], kind: EmbedKind = "passage") -> list[list[float]]:
        if not texts:
            return []

        task = task_for(kind)
        keys = [content_key(model, task, text) for text in texts]
        results: list[list[float] | None] = [cache.get(key) for key in keys]

        # Group misses by key so a batch containing the same text twice costs one call.
        missing: dict[str, list[int]] = {}
        for position, (key, value) in enumerate(zip(keys, results, strict=True)):
            if value is None:
                missing.setdefault(key, []).append(position)

        if missing:
            unique_texts = [texts[positions[0]] for positions in missing.values()]
            fresh = await inner(unique_texts, kind)
            if len(fresh) != len(unique_texts):
                raise RuntimeError(
                    f"embedder returned {len(fresh)} vectors for {len(unique_texts)} texts"
                )
            for key, vector in zip(missing, fresh, strict=True):
                cache.set(key, vector)
                for position in missing[key]:
                    results[position] = vector

        if any(value is None for value in results):
            raise RuntimeError("cache wrapper produced a gap")
        return results  # type: ignore[return-value]

    return embedder
