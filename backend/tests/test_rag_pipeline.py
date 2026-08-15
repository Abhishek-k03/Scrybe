"""Stage assembly: one config in, one working retrieval path out."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.config import (
    ChromaIndexConfig,
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    HybridRetrieveConfig,
    JinaEmbedConfig,
    MemoryIndexConfig,
    MmrRerankConfig,
    NoopRerankConfig,
    PipelineConfig,
    SentenceChunkConfig,
)
from app.rag.pipeline import IndexReport, _with_top_k, build_pipeline
from app.rag.types import Document

DOCS = [
    Document.create(
        "python.txt",
        "Python uses reference counting to free memory. A cycle detector reclaims the rest. "
        "The global interpreter lock serialises bytecode execution across threads.",
    ),
    Document.create(
        "guido.txt",
        "Guido van Rossum created Python and released version 0.9.0 in 1991. "
        "He served as benevolent dictator for life until 2018.",
    ),
    Document.create(
        "numpy.txt",
        "NumPy provides fast multidimensional arrays. Broadcasting avoids explicit loops. "
        "Django is unrelated: it is a web framework with an ORM.",
    ),
]


def offline_config(**overrides) -> PipelineConfig:
    base = {
        "chunk": FixedCharChunkConfig(chunk_size=120, overlap=20),
        "embed": FakeEmbedConfig(dimensions=256),
        "index": MemoryIndexConfig(),
        "retrieve": DenseRetrieveConfig(top_k=3),
        "rerank": NoopRerankConfig(),
    }
    return PipelineConfig(**{**base, **overrides})


@pytest.fixture
async def indexed():
    pipeline = build_pipeline(offline_config())
    await pipeline.index_documents(DOCS)
    return pipeline


# --------------------------------------------------------------------------------------
# assembly
# --------------------------------------------------------------------------------------


def test_every_stage_comes_from_the_config() -> None:
    pipeline = build_pipeline(offline_config(chunk=SentenceChunkConfig(max_chars=100)))
    chunks = pipeline.chunk(DOCS[0])

    assert pipeline.config.chunk.kind == "sentence"
    assert all(len(chunk.text) <= 100 for chunk in chunks)


def test_an_injected_index_wins_over_the_configured_one(tmp_path: Path) -> None:
    """The app already holds an open handle; rebuilding one per request would be waste."""
    from app.rag.index.memory import MemoryIndex

    existing = MemoryIndex(MemoryIndexConfig())
    config = offline_config(index=ChromaIndexConfig(path=str(tmp_path / "unused")))
    pipeline = build_pipeline(config, index=existing)

    assert pipeline.index is existing
    assert not (tmp_path / "unused").exists(), "the configured index was built anyway"


def test_the_api_key_never_enters_the_config() -> None:
    pipeline = build_pipeline(offline_config(embed=JinaEmbedConfig()), api_key="secret-key")
    assert "secret-key" not in pipeline.config.to_json()


def test_mmr_makes_the_pipeline_request_embeddings() -> None:
    assert build_pipeline(offline_config(rerank=MmrRerankConfig())).needs_embeddings
    assert not build_pipeline(offline_config()).needs_embeddings


# --------------------------------------------------------------------------------------
# indexing
# --------------------------------------------------------------------------------------


async def test_indexing_reports_what_it_did(indexed) -> None:
    assert indexed.index.count() > len(DOCS), "documents should split into several chunks"


async def test_reindexing_the_same_documents_is_a_noop(indexed) -> None:
    before = indexed.index.count()
    report = await indexed.index_documents(DOCS)

    assert report == IndexReport(documents_indexed=0, documents_skipped=3, chunks_added=0)
    assert indexed.index.count() == before


async def test_a_changed_document_is_indexed_as_a_new_one(indexed) -> None:
    before = indexed.index.count()
    edited = Document.create("python.txt", "Python uses reference counting. Edited.")

    report = await indexed.index_documents([edited])

    assert report.documents_indexed == 1
    assert indexed.index.count() > before


async def test_skip_existing_can_be_turned_off() -> None:
    pipeline = build_pipeline(offline_config())
    await pipeline.index_documents(DOCS[:1])
    report = await pipeline.index_documents(DOCS[:1], skip_existing=False)

    assert report.documents_indexed == 1


async def test_an_empty_document_is_skipped_not_stored() -> None:
    pipeline = build_pipeline(offline_config())
    report = await pipeline.index_documents([Document.create("blank.txt", "   \n\t ")])

    assert report == IndexReport(documents_indexed=0, documents_skipped=1, chunks_added=0)
    assert pipeline.index.count() == 0


async def test_indexing_nothing_is_an_empty_report() -> None:
    assert await build_pipeline(offline_config()).index_documents([]) == IndexReport()


# --------------------------------------------------------------------------------------
# retrieval
# --------------------------------------------------------------------------------------


async def test_retrieve_returns_top_k_from_the_config(indexed) -> None:
    assert len(await indexed.retrieve("how does python free memory")) == 3


async def test_retrieve_finds_the_relevant_document(indexed) -> None:
    result = await indexed.retrieve("who created python", top_k=3)
    assert "guido.txt" in [hit.chunk.doc_label for hit in result.hits]


async def test_retrieve_carries_the_query_embedding(indexed) -> None:
    """The vector map needs it; re-embedding would be a second billed call."""
    result = await indexed.retrieve("reference counting")
    assert result.query_embedding is not None
    assert len(result.query_embedding) == 256


async def test_blank_query_returns_nothing(indexed) -> None:
    assert len(await indexed.retrieve("   ")) == 0


async def test_retrieval_on_an_empty_index_returns_nothing() -> None:
    assert len(await build_pipeline(offline_config()).retrieve("anything")) == 0


async def test_top_k_override_widens_the_result(indexed) -> None:
    assert len(await indexed.retrieve("python", top_k=6)) == 6


async def test_top_k_override_does_not_mutate_the_config(indexed) -> None:
    await indexed.retrieve("python", top_k=6)
    assert indexed.config.retrieve.top_k == 3


async def test_repeated_retrieval_is_deterministic(indexed) -> None:
    first = await indexed.retrieve("global interpreter lock")
    for _ in range(3):
        assert await indexed.retrieve("global interpreter lock") == first


async def test_hybrid_retrieval_assembles(indexed) -> None:
    pipeline = build_pipeline(offline_config(retrieve=HybridRetrieveConfig(top_k=3)))
    await pipeline.index_documents(DOCS)

    result = await pipeline.retrieve("Django ORM")
    assert any("Django" in hit.chunk.text for hit in result.hits)


async def test_the_bm25_arm_follows_new_documents() -> None:
    """A retriever cached past an ingest would score against a stale corpus."""
    pipeline = build_pipeline(offline_config(retrieve=HybridRetrieveConfig(top_k=3)))
    await pipeline.index_documents(DOCS[:1])
    await pipeline.retrieve("Django ORM")

    await pipeline.index_documents(DOCS[2:])
    result = await pipeline.retrieve("Django ORM")

    assert any("Django" in hit.chunk.text for hit in result.hits)


async def test_mmr_reranking_runs_end_to_end() -> None:
    pipeline = build_pipeline(
        offline_config(
            retrieve=DenseRetrieveConfig(top_k=3, fetch_k=8),
            rerank=MmrRerankConfig(lambda_mult=0.3, top_k=3),
        )
    )
    await pipeline.index_documents(DOCS)

    result = await pipeline.retrieve("python memory management")
    assert len(result) == 3


async def test_a_reranker_cannot_return_more_than_top_k() -> None:
    pipeline = build_pipeline(offline_config(retrieve=DenseRetrieveConfig(top_k=2, fetch_k=8)))
    await pipeline.index_documents(DOCS)

    assert len(await pipeline.retrieve("python")) == 2


# --------------------------------------------------------------------------------------
# top_k override
# --------------------------------------------------------------------------------------


def test_override_returns_the_same_object_when_unchanged() -> None:
    config = DenseRetrieveConfig(top_k=5)
    assert _with_top_k(config, 5) is config


def test_override_widens_a_candidate_pool_that_is_now_too_narrow() -> None:
    """fetch_k below top_k cannot fill the result, and the config forbids it."""
    assert _with_top_k(DenseRetrieveConfig(top_k=5, fetch_k=10), 20).fetch_k == 20


def test_override_leaves_a_wide_enough_pool_alone() -> None:
    assert _with_top_k(DenseRetrieveConfig(top_k=5, fetch_k=50), 10).fetch_k == 50


def test_override_preserves_the_other_settings() -> None:
    config = HybridRetrieveConfig(top_k=5, rrf_k=17, bm25_k1=1.9, score_threshold=0.2)
    widened = _with_top_k(config, 9)

    assert (widened.rrf_k, widened.bm25_k1, widened.score_threshold) == (17, 1.9, 0.2)
    assert widened.kind == "hybrid"
