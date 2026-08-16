"""Retrieval and reranking."""

from __future__ import annotations

import pytest

from app.rag import registry
from app.rag.config import (
    DenseRetrieveConfig,
    FakeEmbedConfig,
    HybridRetrieveConfig,
    MemoryIndexConfig,
    MmrRerankConfig,
    NoopRerankConfig,
)
from app.rag.embed.fake import embed_texts
from app.rag.index.memory import MemoryIndex
from app.rag.rerank.mmr import MissingEmbeddingError, cosine, rerank_mmr
from app.rag.retrieve.bm25 import build_bm25, rank, tokenize
from app.rag.retrieve.dense import apply_threshold, retrieve_dense
from app.rag.retrieve.hybrid import reciprocal_rank_fusion, retrieve_hybrid
from app.rag.types import Chunk, Hit, RetrievalResult

EMBED = FakeEmbedConfig(dimensions=128)

TEXTS = [
    "Python uses reference counting to free memory automatically",
    "The global interpreter lock prevents true thread parallelism in CPython",
    "NumPy provides fast multidimensional array operations for numerical work",
    "Django is a high level web framework that ships with an ORM",
    "Guido van Rossum created Python and released version 0.9.0 in 1991",
    "Reference counting cannot reclaim reference cycles on its own",
]


def make_chunks(texts: list[str] = TEXTS) -> list[Chunk]:
    return [
        Chunk(
            doc_id="doc1",
            doc_label="doc1.txt",
            source_type="fixture",
            chunk_index=i,
            text=text,
            start_char=i * 100,
            end_char=i * 100 + len(text),
        )
        for i, text in enumerate(texts)
    ]


async def fake_embedder(texts, kind="passage"):
    return embed_texts(list(texts), EMBED)


@pytest.fixture
def index() -> MemoryIndex:
    idx = MemoryIndex(MemoryIndexConfig())
    chunks = make_chunks()
    idx.add(chunks, embed_texts([c.text for c in chunks], EMBED))
    return idx


def make_hit(index_: int, score: float, embedding: list[float] | None = None) -> Hit:
    return Hit(
        chunk=make_chunks(["placeholder text"] * (index_ + 1))[index_],
        score=score,
        embedding=tuple(embedding) if embedding is not None else None,
    )


# --------------------------------------------------------------------------------------
# dense
# --------------------------------------------------------------------------------------


async def test_dense_returns_top_k(index: MemoryIndex) -> None:
    result = await retrieve_dense("python memory", DenseRetrieveConfig(top_k=3), fake_embedder, index)
    assert len(result) == 3


async def test_dense_returns_fetch_k_candidates_when_set(index: MemoryIndex) -> None:
    """fetch_k widens the pool for the reranker; trimming happens later."""
    config = DenseRetrieveConfig(top_k=2, fetch_k=5)
    result = await retrieve_dense("python", config, fake_embedder, index)
    assert len(result) == 5


async def test_dense_orders_by_score(index: MemoryIndex) -> None:
    result = await retrieve_dense("reference counting", DenseRetrieveConfig(top_k=6), fake_embedder, index)
    scores = [hit.score for hit in result.hits]
    assert scores == sorted(scores, reverse=True)


async def test_dense_on_blank_query_returns_nothing(index: MemoryIndex) -> None:
    for query in ["", "   ", "\n\t"]:
        result = await retrieve_dense(query, DenseRetrieveConfig(), fake_embedder, index)
        assert len(result) == 0


async def test_dense_can_populate_embeddings(index: MemoryIndex) -> None:
    result = await retrieve_dense(
        "python", DenseRetrieveConfig(top_k=2), fake_embedder, index, with_embeddings=True
    )
    assert all(hit.embedding is not None for hit in result.hits)


async def test_dense_omits_embeddings_by_default(index: MemoryIndex) -> None:
    result = await retrieve_dense("python", DenseRetrieveConfig(top_k=2), fake_embedder, index)
    assert all(hit.embedding is None for hit in result.hits)


# --------------------------------------------------------------------------------------
# threshold — the abstention behavior the old retriever could not express
# --------------------------------------------------------------------------------------


def test_threshold_of_none_keeps_everything() -> None:
    hits = [make_hit(0, 0.9), make_hit(1, 0.1)]
    assert apply_threshold(hits, None) == hits


def test_threshold_drops_low_scores() -> None:
    hits = [make_hit(0, 0.9), make_hit(1, 0.4), make_hit(2, 0.1)]
    assert [h.score for h in apply_threshold(hits, 0.5)] == [0.9]


def test_threshold_is_inclusive_at_the_boundary() -> None:
    assert len(apply_threshold([make_hit(0, 0.5)], 0.5)) == 1


async def test_impossible_threshold_returns_an_empty_result(index: MemoryIndex) -> None:
    """Retrieval must be able to say nothing matched."""
    config = DenseRetrieveConfig(top_k=5, score_threshold=0.99)
    result = await retrieve_dense("entirely unrelated tractor farming", config, fake_embedder, index)
    assert len(result) == 0
    assert result.hits == ()


# --------------------------------------------------------------------------------------
# bm25
# --------------------------------------------------------------------------------------


def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, World! Python 3.13") == ["hello", "world", "python", "3", "13"]


def test_bm25_finds_the_exact_term() -> None:
    bm25 = build_bm25(make_chunks())
    top = rank(bm25, "NumPy multidimensional", top_k=1, k1=1.5, b=0.75)
    assert "NumPy" in bm25.chunks[top[0][0]].text


def test_bm25_returns_nothing_for_absent_terms() -> None:
    bm25 = build_bm25(make_chunks())
    assert rank(bm25, "kubernetes helm chart", top_k=5, k1=1.5, b=0.75) == []


def test_bm25_scores_are_positive_and_descending() -> None:
    bm25 = build_bm25(make_chunks())
    scores = [score for _, score in rank(bm25, "reference counting python", 5, 1.5, 0.75)]
    assert all(s > 0 for s in scores)
    assert scores == sorted(scores, reverse=True)


def test_bm25_on_an_empty_corpus() -> None:
    bm25 = build_bm25([])
    assert rank(bm25, "anything", 5, 1.5, 0.75) == []


def test_bm25_rewards_rarer_terms() -> None:
    """'python' appears in several chunks; 'django' in one."""
    bm25 = build_bm25(make_chunks())
    common = dict(rank(bm25, "python", 10, 1.5, 0.75))
    rare = dict(rank(bm25, "django", 10, 1.5, 0.75))
    assert max(rare.values()) > max(common.values())


# --------------------------------------------------------------------------------------
# hybrid
# --------------------------------------------------------------------------------------


def test_rrf_rewards_appearing_in_both_rankings() -> None:
    fused = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], rrf_k=60)
    assert fused["a"] == fused["b"]


def test_rrf_ranks_earlier_positions_higher() -> None:
    fused = reciprocal_rank_fusion([["a", "b", "c"]], rrf_k=60)
    assert fused["a"] > fused["b"] > fused["c"]


def test_rrf_of_no_rankings_is_empty() -> None:
    assert reciprocal_rank_fusion([], rrf_k=60) == {}


async def test_hybrid_surfaces_a_lexical_match(index: MemoryIndex) -> None:
    bm25 = build_bm25(index.chunks())
    config = HybridRetrieveConfig(top_k=3)
    result = await retrieve_hybrid("Django ORM", config, fake_embedder, index, bm25)
    assert any("Django" in hit.chunk.text for hit in result.hits)


async def test_hybrid_returns_a_union_of_both_arms(index: MemoryIndex) -> None:
    bm25 = build_bm25(index.chunks())
    config = HybridRetrieveConfig(top_k=6)
    result = await retrieve_hybrid("global interpreter lock", config, fake_embedder, index, bm25)

    ids = [hit.chunk.chunk_id for hit in result.hits]
    assert len(ids) == len(set(ids)), "fusion must not duplicate a chunk"


async def test_hybrid_on_blank_query_returns_nothing(index: MemoryIndex) -> None:
    bm25 = build_bm25(index.chunks())
    result = await retrieve_hybrid("  ", HybridRetrieveConfig(), fake_embedder, index, bm25)
    assert len(result) == 0


# --------------------------------------------------------------------------------------
# rerank
# --------------------------------------------------------------------------------------


async def test_noop_preserves_order_exactly() -> None:
    hits = [make_hit(0, 0.9), make_hit(1, 0.5), make_hit(2, 0.1)]
    reranker = registry.build("rerank", NoopRerankConfig())
    assert await reranker(RetrievalResult(query="q", hits=tuple(hits))) == hits


async def test_a_reranker_receives_the_query_text() -> None:
    """Whatever a stage does with it, the query has to reach it. It used not to."""
    seen: list[str] = []

    async def reranker(result: RetrievalResult) -> list[Hit]:
        seen.append(result.query)
        return list(result.hits)

    await reranker(RetrievalResult(query="why is the GIL there", hits=(make_hit(0, 0.9),)))
    assert seen == ["why is the GIL there"]


async def test_mmr_through_the_registry_reads_hits_off_the_result() -> None:
    hits = [
        make_hit(0, 0.90, [1.0, 0.0]),
        make_hit(1, 0.89, [1.0, 0.0]),
        make_hit(2, 0.40, [0.0, 1.0]),
    ]
    reranker = registry.build("rerank", MmrRerankConfig(lambda_mult=0.0, top_k=2))
    result = await reranker(RetrievalResult(query="q", hits=tuple(hits)))
    assert [h.chunk.chunk_index for h in result] == [0, 2]


def test_mmr_with_lambda_one_reproduces_input_order() -> None:
    """Pure relevance must not reorder anything."""
    hits = [
        make_hit(0, 0.9, [1.0, 0.0]),
        make_hit(1, 0.8, [1.0, 0.0]),
        make_hit(2, 0.7, [0.0, 1.0]),
    ]
    result = rerank_mmr(hits, MmrRerankConfig(lambda_mult=1.0, top_k=3))
    assert result == hits


def test_mmr_with_lambda_zero_prefers_diversity() -> None:
    """The near-duplicate of the first pick must lose to the orthogonal candidate."""
    hits = [
        make_hit(0, 0.90, [1.0, 0.0]),
        make_hit(1, 0.89, [1.0, 0.0]),
        make_hit(2, 0.40, [0.0, 1.0]),
    ]
    result = rerank_mmr(hits, MmrRerankConfig(lambda_mult=0.0, top_k=2))
    assert [h.chunk.chunk_index for h in result] == [0, 2]


def test_mmr_truncates_to_top_k() -> None:
    hits = [make_hit(i, 1.0 - i / 10, [1.0, float(i)]) for i in range(6)]
    assert len(rerank_mmr(hits, MmrRerankConfig(top_k=3))) == 3


def test_mmr_top_k_larger_than_candidates_returns_all() -> None:
    hits = [make_hit(i, 1.0 - i / 10, [1.0, float(i)]) for i in range(2)]
    assert len(rerank_mmr(hits, MmrRerankConfig(top_k=10))) == 2


def test_mmr_on_no_candidates() -> None:
    assert rerank_mmr([], MmrRerankConfig()) == []


def test_mmr_never_repeats_a_candidate() -> None:
    hits = [make_hit(i, 0.5, [1.0, 0.0]) for i in range(4)]
    result = rerank_mmr(hits, MmrRerankConfig(lambda_mult=0.5, top_k=4))
    assert len({h.chunk.chunk_id for h in result}) == 4


def test_mmr_without_embeddings_fails_loudly() -> None:
    """Silently falling back to relevance order would hide a wiring bug."""
    with pytest.raises(MissingEmbeddingError, match="with_embeddings=True"):
        rerank_mmr([make_hit(0, 0.9)], MmrRerankConfig())


def test_cosine_of_identical_vectors_is_one() -> None:
    assert abs(cosine([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-12


def test_cosine_of_orthogonal_vectors_is_zero() -> None:
    assert abs(cosine([1.0, 0.0], [0.0, 1.0])) < 1e-12


def test_cosine_with_a_zero_vector_is_zero() -> None:
    assert cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# --------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------


def test_retrieve_and_rerank_kinds_are_registered() -> None:
    assert registry.registered_kinds("retrieve") == ("dense", "hybrid")
    assert registry.registered_kinds("rerank") == ("jina_rerank", "mmr", "noop")


async def test_registry_builds_a_dense_retriever(index: MemoryIndex) -> None:
    retriever = registry.build(
        "retrieve", DenseRetrieveConfig(top_k=2), embedder=fake_embedder, index=index
    )
    assert len(await retriever("python")) == 2


async def test_registry_builds_a_hybrid_retriever(index: MemoryIndex) -> None:
    retriever = registry.build(
        "retrieve", HybridRetrieveConfig(top_k=3), embedder=fake_embedder, index=index
    )
    assert len(await retriever("django orm")) > 0
