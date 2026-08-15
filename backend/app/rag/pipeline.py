"""Assembles the stages a `PipelineConfig` names into one object.

`build_pipeline(config)` is the only place the six stages are wired together, so an eval
sweep and the running app exercise the same retrieval path. Secrets are passed here as
arguments, never through the config, which gets serialised into result artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag import registry
from app.rag.config import PipelineConfig, RetrieveConfig
from app.rag.protocols import Chunker, Embedder, Retriever, VectorIndex
from app.rag.types import Chunk, Document, RetrievalResult


@dataclass(frozen=True)
class IndexReport:
    """What one `index_documents` call actually did."""

    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_added: int = 0


def _with_top_k(config: RetrieveConfig, top_k: int) -> RetrieveConfig:
    """Copy a retrieve config with a different `top_k`, re-running its validators."""
    if top_k == config.top_k:
        return config
    data = config.model_dump()
    data["top_k"] = top_k
    # A candidate pool narrower than the requested k could not fill it.
    if data.get("fetch_k") is not None and data["fetch_k"] < top_k:
        data["fetch_k"] = top_k
    return type(config).model_validate(data)


class Pipeline:
    def __init__(
        self,
        config: PipelineConfig,
        *,
        chunker: Chunker,
        embedder: Embedder,
        index: VectorIndex,
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.embedder = embedder
        self.index = index
        self._retriever: Retriever | None = None
        self._retriever_key: tuple[object, int] | None = None

    @property
    def needs_embeddings(self) -> bool:
        """MMR compares candidates pairwise, so it needs the vectors carried on each hit."""
        return self.config.rerank.kind == "mmr"

    def chunk(self, doc: Document) -> list[Chunk]:
        return self.chunker(doc)

    async def index_documents(
        self, docs: Sequence[Document], *, skip_existing: bool = True
    ) -> IndexReport:
        """Chunk, embed and store documents the index does not already hold.

        Document ids are content hashes, so skipping what is already present makes
        re-indexing the same corpus a no-op rather than a second copy.
        """
        indexed = skipped = added = 0

        for doc in docs:
            if skip_existing and self.index.has_doc(doc.doc_id):
                skipped += 1
                continue
            chunks = self.chunker(doc)
            if not chunks:
                skipped += 1
                continue
            vectors = await self.embedder([chunk.text for chunk in chunks], "passage")
            added += self.index.add(chunks, vectors)
            indexed += 1

        return IndexReport(
            documents_indexed=indexed,
            documents_skipped=skipped,
            chunks_added=added,
        )

    def _retriever_for(self, config: RetrieveConfig) -> Retriever:
        # Rebuilt when the row count moves because hybrid's BM25 statistics are computed
        # from the stored chunks. An equal-sized delete-then-add between calls would go
        # unnoticed; re-indexing in place does not happen in this app.
        key = (config, self.index.count())
        if self._retriever_key != key:
            self._retriever = registry.build(
                "retrieve",
                config,
                embedder=self.embedder,
                index=self.index,
                with_embeddings=self.needs_embeddings,
            )
            self._retriever_key = key
        assert self._retriever is not None
        return self._retriever

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        retrieve_config = self.config.retrieve
        if top_k is not None:
            retrieve_config = _with_top_k(retrieve_config, top_k)

        result = await self._retriever_for(retrieve_config)(query)

        rerank_config = self.config.rerank
        if top_k is not None and hasattr(rerank_config, "top_k"):
            rerank_config = rerank_config.model_copy(update={"top_k": top_k})
        hits = registry.build("rerank", rerank_config)(result.query_embedding, result.hits)

        return result.model_copy(update={"hits": tuple(hits[: retrieve_config.top_k])})


def build_pipeline(
    config: PipelineConfig,
    *,
    api_key: str = "",
    index: VectorIndex | None = None,
) -> Pipeline:
    """Build every stage from `config`.

    `index` overrides the configured one for callers that already hold an open handle;
    everything else comes from the config alone.
    """
    embed_kwargs = {"api_key": api_key} if config.embed.kind == "jina" else {}
    return Pipeline(
        config,
        chunker=registry.build("chunk", config.chunk),
        embedder=registry.build("embed", config.embed, **embed_kwargs),
        index=index if index is not None else registry.build("index", config.index),
    )
