"""Rerank stage. Importing this package registers every variant."""

from app.rag.rerank import mmr, noop

__all__ = ["mmr", "noop"]
