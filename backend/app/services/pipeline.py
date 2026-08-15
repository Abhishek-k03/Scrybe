"""The app's composition root: which `PipelineConfig` the server runs, and the singletons.

Everything else under `services/` is a formatting shim over what this module builds, so the
server and an offline eval exercise the same retrieval code. Settings are read when the
singletons are built — call `reset()` after changing them.
"""

from __future__ import annotations

import logging
from threading import RLock

from app.core.config import settings
from app.rag.config import (
    ChromaIndexConfig,
    DenseRetrieveConfig,
    FixedCharChunkConfig,
    JinaEmbedConfig,
    NoopRerankConfig,
    PipelineConfig,
)
from app.rag.index.chroma import ChromaIndex
from app.rag.pipeline import Pipeline, build_pipeline

log = logging.getLogger("scrybe.pipeline")

DEFAULT_TOP_K = 5

_index: ChromaIndex | None = None
_pipeline: Pipeline | None = None
# Reentrant: get() holds it while calling get_index().
_lock = RLock()


def default_config() -> PipelineConfig:
    """The production configuration.

    Chunking stays at 800/150 and reranking at noop: those change when an eval artifact
    says they should, not before.
    """
    return PipelineConfig(
        chunk=FixedCharChunkConfig(),
        embed=JinaEmbedConfig(),
        index=ChromaIndexConfig(
            path=settings.CHROMA_PATH,
            collection=settings.COLLECTION_NAME,
            read_only=False,
        ),
        retrieve=DenseRetrieveConfig(top_k=DEFAULT_TOP_K),
        rerank=NoopRerankConfig(),
    )


def get_index() -> ChromaIndex:
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                config = default_config().index
                _index = ChromaIndex(config)
                log.info("ChromaDB ready at %s (collection=%s)", config.path, config.collection)
    return _index


def get() -> Pipeline:
    global _pipeline
    if _pipeline is None:
        with _lock:
            if _pipeline is None:
                _pipeline = build_pipeline(
                    default_config(),
                    api_key=settings.JINA_API_KEY,
                    index=get_index(),
                )
    return _pipeline


def reset() -> None:
    """Drop the singletons so the next call re-reads settings."""
    global _index, _pipeline
    _index = None
    _pipeline = None
