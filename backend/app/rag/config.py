"""Configuration for every pipeline stage.

One `PipelineConfig` fully determines a run and serialises to JSON, so an eval artifact can
record exactly what produced its numbers. `extra="forbid"` makes a mistyped key in a sweep
file fail loudly instead of silently running the default and being reported as the variant.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StageConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


# --------------------------------------------------------------------------------------
# chunk
# --------------------------------------------------------------------------------------


class FixedCharChunkConfig(StageConfig):
    """Fixed-width character windows. Matches the original chunker at 800/150."""

    kind: Literal["fixed_char"] = "fixed_char"
    chunk_size: int = Field(default=800, gt=0)
    overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def _overlap_fits(self) -> FixedCharChunkConfig:
        if self.chunk_size <= self.overlap:
            raise ValueError("chunk_size must be greater than overlap")
        return self

    @property
    def stride(self) -> int:
        return self.chunk_size - self.overlap


class SentenceChunkConfig(StageConfig):
    """Pack whole sentences up to a character budget."""

    kind: Literal["sentence"] = "sentence"
    max_chars: int = Field(default=800, gt=0)
    overlap_sentences: int = Field(default=1, ge=0)


class TokenChunkConfig(StageConfig):
    kind: Literal["token"] = "token"
    max_tokens: int = Field(default=200, gt=0)
    overlap_tokens: int = Field(default=40, ge=0)
    encoding: str = "cl100k_base"

    @model_validator(mode="after")
    def _overlap_fits(self) -> TokenChunkConfig:
        if self.max_tokens <= self.overlap_tokens:
            raise ValueError("max_tokens must be greater than overlap_tokens")
        return self


ChunkConfig = Annotated[
    FixedCharChunkConfig | SentenceChunkConfig | TokenChunkConfig,
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------------------
# embed
# --------------------------------------------------------------------------------------


class JinaEmbedConfig(StageConfig):
    kind: Literal["jina"] = "jina"
    model: str = "jina-embeddings-v3"
    passage_task: str = "retrieval.passage"
    query_task: str = "retrieval.query"
    batch_size: int = Field(default=32, gt=0)
    dimensions: int = Field(default=1024, gt=0)
    # No default: a relative path would resolve against the caller's working directory and
    # scatter caches wherever a script happened to be run from.
    cache_dir: str | None = None
    max_retries: int = Field(default=3, ge=0)


class FakeEmbedConfig(StageConfig):
    """Deterministic vectors derived from a text hash, for tests and offline runs."""

    kind: Literal["fake"] = "fake"
    dimensions: int = Field(default=64, gt=0)
    seed: int = 0


EmbedConfig = Annotated[JinaEmbedConfig | FakeEmbedConfig, Field(discriminator="kind")]


# --------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------


class ChromaIndexConfig(StageConfig):
    """Persistent Chroma index.

    `path` has no default and `read_only` defaults to True: addressing an index requires
    naming it, and writing to one requires saying so.
    """

    kind: Literal["chroma"] = "chroma"
    path: str
    collection: str = "scrybe"
    space: Literal["cosine"] = "cosine"
    read_only: bool = True


class MemoryIndexConfig(StageConfig):
    """Exact in-process search. No files, no ANN approximation, fully deterministic."""

    kind: Literal["memory"] = "memory"
    space: Literal["cosine"] = "cosine"


IndexConfig = Annotated[ChromaIndexConfig | MemoryIndexConfig, Field(discriminator="kind")]


# --------------------------------------------------------------------------------------
# retrieve
# --------------------------------------------------------------------------------------


class DenseRetrieveConfig(StageConfig):
    kind: Literal["dense"] = "dense"
    top_k: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(default=None, gt=0)
    score_threshold: float | None = None

    @model_validator(mode="after")
    def _fetch_covers_top_k(self) -> DenseRetrieveConfig:
        if self.fetch_k is not None and self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be at least top_k")
        return self


class HybridRetrieveConfig(StageConfig):
    """Dense and BM25 candidates fused by reciprocal rank."""

    kind: Literal["hybrid"] = "hybrid"
    top_k: int = Field(default=5, gt=0)
    fetch_k: int | None = Field(default=None, gt=0)
    score_threshold: float | None = None
    rrf_k: int = Field(default=60, gt=0)
    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _fetch_covers_top_k(self) -> HybridRetrieveConfig:
        if self.fetch_k is not None and self.fetch_k < self.top_k:
            raise ValueError("fetch_k must be at least top_k")
        return self


RetrieveConfig = Annotated[
    DenseRetrieveConfig | HybridRetrieveConfig, Field(discriminator="kind")
]


# --------------------------------------------------------------------------------------
# rerank
# --------------------------------------------------------------------------------------


class NoopRerankConfig(StageConfig):
    kind: Literal["noop"] = "noop"


class MmrRerankConfig(StageConfig):
    """Maximal marginal relevance. `lambda_mult=1.0` is pure relevance, 0.0 pure diversity."""

    kind: Literal["mmr"] = "mmr"
    lambda_mult: float = Field(default=0.5, ge=0.0, le=1.0)
    top_k: int = Field(default=5, gt=0)


RerankConfig = Annotated[NoopRerankConfig | MmrRerankConfig, Field(discriminator="kind")]


# --------------------------------------------------------------------------------------
# pipeline
# --------------------------------------------------------------------------------------


class PipelineConfig(BaseModel):
    """Defaults are the safe offline combination: no network, no disk, no live index."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk: ChunkConfig = FixedCharChunkConfig()
    embed: EmbedConfig = FakeEmbedConfig()
    index: IndexConfig = MemoryIndexConfig()
    retrieve: RetrieveConfig = DenseRetrieveConfig()
    rerank: RerankConfig = NoopRerankConfig()

    @classmethod
    def from_json(cls, raw: str) -> PipelineConfig:
        return cls.model_validate_json(raw)

    @classmethod
    def from_file(cls, path: str | Path) -> PipelineConfig:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.model_dump(mode="json"), indent=indent, sort_keys=True)
