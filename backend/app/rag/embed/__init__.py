"""Embed stage. Importing this package registers every variant."""

from app.rag.embed import fake, jina
from app.rag.embed.cached import cached_embedder

__all__ = ["cached_embedder", "fake", "jina"]
