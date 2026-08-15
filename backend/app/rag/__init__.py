"""Standalone retrieval pipeline: ingest -> chunk -> embed -> index -> retrieve -> rerank.

Importable without booting FastAPI. Nothing in this package reads global configuration —
every stage takes its config as an argument, so multiple pipelines can coexist in one
process and point at different indexes.
"""

from app.rag import chunk as _chunk  # noqa: F401  registers the chunk variants
from app.rag import embed as _embed  # noqa: F401  registers the embed variants
from app.rag import index as _index  # noqa: F401  registers the index variants
from app.rag import ingest as _ingest  # noqa: F401
from app.rag import rerank as _rerank  # noqa: F401  registers the rerank variants
from app.rag import retrieve as _retrieve  # noqa: F401  registers the retrieve variants
from app.rag.config import PipelineConfig
from app.rag.pipeline import IndexReport, Pipeline, build_pipeline
from app.rag.types import Chunk, Document, Hit, RetrievalResult, make_doc_id

__all__ = [
    "Chunk",
    "Document",
    "Hit",
    "IndexReport",
    "Pipeline",
    "PipelineConfig",
    "RetrievalResult",
    "build_pipeline",
    "make_doc_id",
]
