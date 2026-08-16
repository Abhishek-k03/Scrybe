"""Assembles the stages a `PipelineConfig` names into one object.

`build_pipeline(config)` is the only place the six stages are wired together, so an eval
sweep and the running app exercise the same retrieval path. Secrets are passed here as
arguments, never through the config, which gets serialised into result artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from app.rag import registry
from app.rag.config import PipelineConfig, RerankConfig, RetrieveConfig
from app.rag.protocols import Chunker, Embedder, Reranker, Retriever, VectorIndex
from app.rag.types import Chunk, Document, RetrievalResult

# Rerank kinds that compare candidates to each other rather than to the query, and so need
# the vectors fetched alongside each hit. Listed here rather than tested for inline so a new
# stage that needs them cannot quietly receive `embedding=None` instead.
RERANK_KINDS_NEEDING_EMBEDDINGS = frozenset({"mmr"})

# Stage variants that call a hosted API and so need a key. Secrets travel as build kwargs,
# never on the config, which gets serialised into every eval artifact.
KINDS_NEEDING_API_KEY = frozenset({"jina", "jina_rerank"})


@dataclass(frozen=True)
class IndexReport:
    """What one `index_documents` call actually did."""

    documents_indexed: int = 0
    documents_skipped: int = 0
    chunks_added: int = 0


def _key_kwargs(config: object, api_key: str) -> dict[str, str]:
    """`{"api_key": ...}` for the stage variants that take one, `{}` for the rest."""
    return {"api_key": api_key} if getattr(config, "kind", None) in KINDS_NEEDING_API_KEY else {}


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
        api_key: str = "",
    ) -> None:
        self.config = config
        self.chunker = chunker
        self.embedder = embedder
        self.index = index
        # Held rather than baked into a reranker at build time: the rerank config is rebuilt
        # per call when `top_k` is overridden, so the key has to outlive any one of them.
        self.api_key = api_key
        self._retriever: Retriever | None = None
        self._retriever_key: tuple[object, int] | None = None
        self._reranker: Reranker | None = None
        self._reranker_key: object | None = None

    @property
    def needs_embeddings(self) -> bool:
        """Whether the rerank stage needs the vector carried on each hit."""
        return self.config.rerank.kind in RERANK_KINDS_NEEDING_EMBEDDINGS

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

    def _reranker_for(self, config: RerankConfig) -> Reranker:
        # Held across calls so a reranker that owns a cache keeps it, rather than starting
        # cold on every query.
        if self._reranker_key != config:
            self._reranker = registry.build("rerank", config, **_key_kwargs(config, self.api_key))
            self._reranker_key = config
        assert self._reranker is not None
        return self._reranker

    async def retrieve(self, query: str, top_k: int | None = None) -> RetrievalResult:
        retrieve_config = self.config.retrieve
        if top_k is not None:
            retrieve_config = _with_top_k(retrieve_config, top_k)

        result = await self._retriever_for(retrieve_config)(query)

        rerank_config = self.config.rerank
        if top_k is not None and hasattr(rerank_config, "top_k"):
            rerank_config = rerank_config.model_copy(update={"top_k": top_k})
        hits = await self._reranker_for(rerank_config)(result)

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
    return Pipeline(
        config,
        chunker=registry.build("chunk", config.chunk),
        embedder=registry.build("embed", config.embed, **_key_kwargs(config.embed, api_key)),
        index=index if index is not None else registry.build("index", config.index),
        api_key=api_key,
    )
