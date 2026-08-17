"""The gold-rank diagnostic and the perfect-reranker ceiling it computes.

The ceiling is an upper bound that gets quoted to justify building a reranker, so the
arithmetic is worth pinning: it must never promise more than `k`, never count a gold chunk
the pool did not reach, and never round a partial recovery up to a win.
"""

from __future__ import annotations

import pytest

from app.rag.config import (
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    MemoryIndexConfig,
    NoopRerankConfig,
    PipelineConfig,
)
from app.rag.pipeline import build_pipeline
from app.rag.types import Document
from evals import gold_rank
from evals.label_schema import GoldSpan, LabelledQuery, LabelSet

# --------------------------------------------------------------------------------------
# finding the ranks
# --------------------------------------------------------------------------------------


def test_ranks_are_one_based() -> None:
    """Rank 1 is the top hit; a 0 here would read as "better than first"."""
    assert gold_rank.ranks_of(["a", "b", "c"], frozenset({"a"})) == [1]


def test_every_relevant_chunk_is_ranked() -> None:
    assert gold_rank.ranks_of(["a", "b", "c"], frozenset({"a", "c"})) == [1, 3]


def test_ranks_come_back_ascending() -> None:
    assert gold_rank.ranks_of(["c", "b", "a"], frozenset({"a", "c"})) == [1, 3]


def test_a_chunk_that_was_never_retrieved_has_no_rank() -> None:
    assert gold_rank.ranks_of(["a", "b"], frozenset({"z"})) == []


# --------------------------------------------------------------------------------------
# the ceiling
# --------------------------------------------------------------------------------------


def test_gold_inside_the_pool_is_recoverable() -> None:
    """Rank 30 is out of reach at fetch_k=20 and in reach at 50. That is the whole point."""
    assert gold_rank.ceiling_at([30], 1, pool=20, k=5) == 0.0
    assert gold_rank.ceiling_at([30], 1, pool=50, k=5) == 1.0


def test_gold_already_in_the_top_k_needs_no_reranker() -> None:
    assert gold_rank.ceiling_at([1, 2], 2, pool=5, k=5) == 1.0


def test_the_ceiling_cannot_exceed_the_cut_off() -> None:
    """Six gold chunks cannot all fit in a top-5 answer, however good the reranker."""
    assert gold_rank.ceiling_at([1, 2, 3, 4, 5, 6], 6, pool=50, k=5) == pytest.approx(5 / 6)


def test_partial_recovery_is_partial() -> None:
    assert gold_rank.ceiling_at([3, 80], 2, pool=50, k=5) == 0.5


def test_a_query_whose_gold_is_beyond_every_pool_scores_zero() -> None:
    assert gold_rank.ceiling_at([400], 1, pool=100, k=5) == 0.0


def test_no_relevant_chunks_is_zero_not_a_crash() -> None:
    assert gold_rank.ceiling_at([], 0, pool=50, k=5) == 0.0


def test_a_wider_pool_never_lowers_the_ceiling() -> None:
    ranks = [2, 17, 63]
    values = [gold_rank.ceiling_at(ranks, 3, pool=p, k=5) for p in gold_rank.POOL_WIDTHS]
    assert values == sorted(values)


# --------------------------------------------------------------------------------------
# the summary
# --------------------------------------------------------------------------------------


def row(ceiling: dict[str, float], ranks: list[int]) -> dict:
    return {"ceiling": ceiling, "gold_ranks": ranks, "n_relevant": len(ranks)}


def test_the_summary_averages_each_pool_width_separately() -> None:
    rows = [
        row({"5": 1.0, "10": 1.0, "20": 1.0, "50": 1.0, "100": 1.0}, [1]),
        row({"5": 0.0, "10": 0.0, "20": 0.0, "50": 1.0, "100": 1.0}, [30]),
    ]
    summary = gold_rank.summarise(rows, 5)

    assert summary["corpus_rank_ceiling"]["5"] == 0.5
    assert summary["corpus_rank_ceiling"]["50"] == 1.0


def test_the_summary_names_the_worst_rank() -> None:
    """One gold chunk at 352 is the difference between "widen the pool" and "fix embeddings"."""
    rows = [
        row({str(p): 0.0 for p in gold_rank.POOL_WIDTHS}, [11, 12]),
        row({str(p): 0.0 for p in gold_rank.POOL_WIDTHS}, [42, 352]),
    ]
    assert gold_rank.summarise(rows, 5)["worst_gold_rank"] == 352


def test_an_empty_run_summarises_without_dividing_by_zero() -> None:
    assert gold_rank.summarise([], 5) == {"n_queries": 0}


# --------------------------------------------------------------------------------------
# end to end
# --------------------------------------------------------------------------------------

DOCS = [
    Document.create(
        "gil.txt",
        "The global interpreter lock means only one thread may execute bytecode at a time.",
    ),
    Document.create("numpy.txt", "NumPy adds support for large multidimensional arrays."),
    Document.create("django.txt", "Django is a free and open-source Python web framework."),
]

LABELS = LabelSet(
    version=1,
    corpus_sha256="0" * 64,
    author="a test",
    queries=[
        LabelledQuery(
            id="q1",
            query="what is the GIL",
            answerable=True,
            gold=[GoldSpan(doc="gil.txt", span="only one thread may execute bytecode")],
        ),
        LabelledQuery(id="q2", query="nothing here", answerable=False),
        LabelledQuery(
            id="q3",
            query="unmatchable",
            answerable=True,
            gold=[GoldSpan(doc="gil.txt", span="a span that is not in the corpus")],
        ),
    ],
)


def offline_config() -> PipelineConfig:
    return PipelineConfig(
        chunk=FixedCharChunkConfig(chunk_size=200, overlap=40),
        embed=FakeEmbedConfig(dimensions=128),
        index=MemoryIndexConfig(),
        retrieve=DenseRetrieveConfig(top_k=3),
        rerank=NoopRerankConfig(),
    )


@pytest.fixture
async def indexed():
    pipeline = build_pipeline(offline_config())
    await pipeline.index_documents(DOCS)
    return pipeline


async def test_unanswerable_queries_are_skipped(indexed) -> None:
    doc_ids = {doc.label: doc.doc_id for doc in DOCS}
    rows = await gold_rank.measure(indexed, LABELS, doc_ids, 5)
    assert "q2" not in {r["id"] for r in rows}


async def test_a_query_with_no_reachable_gold_is_skipped(indexed) -> None:
    """Its span is not in any chunk, so there is no rank to report — same rule as run.py."""
    doc_ids = {doc.label: doc.doc_id for doc in DOCS}
    rows = await gold_rank.measure(indexed, LABELS, doc_ids, 5)
    assert "q3" not in {r["id"] for r in rows}


async def test_ranking_covers_the_whole_corpus_not_the_configured_top_k(indexed) -> None:
    """top_k=3 in the config must not cap the diagnostic at 3, or deep gold reads as absent."""
    doc_ids = {doc.label: doc.doc_id for doc in DOCS}
    rows = await gold_rank.measure(indexed, LABELS, doc_ids, 5)

    assert rows[0]["n_unranked"] == 0, "a gold chunk in the index was reported as unretrieved"


async def test_every_pool_width_gets_a_ceiling(indexed) -> None:
    doc_ids = {doc.label: doc.doc_id for doc in DOCS}
    rows = await gold_rank.measure(indexed, LABELS, doc_ids, 5)
    assert set(rows[0]["ceiling"]) == {str(p) for p in gold_rank.POOL_WIDTHS}


async def test_the_report_only_lists_queries_a_reranker_could_help(indexed) -> None:
    doc_ids = {doc.label: doc.doc_id for doc in DOCS}
    rows = await gold_rank.measure(indexed, LABELS, doc_ids, 5)
    artifact = {
        "run": {"n_chunks": 3, "k": 5},
        "summary": gold_rank.summarise(rows, 5),
        "per_query": rows,
    }
    text = gold_rank.format_report(artifact, 5)

    # q1's gold is already at rank 1, so there is nothing for a reranker to recover.
    assert "q1" not in text
    assert "perfect reranker" in text


def test_the_report_says_the_ceiling_is_a_bound_not_a_target() -> None:
    artifact = {
        "run": {"n_chunks": 455, "k": 5},
        "summary": gold_rank.summarise(
            [row({str(p): 0.5 for p in gold_rank.POOL_WIDTHS}, [7])], 5
        ),
        "per_query": [
            {"id": "q1", "gold_ranks": [7], "n_relevant": 1, "recall@5": 0.0, "ceiling": {}}
        ],
    }
    assert "upper bound" in gold_rank.format_report(artifact, 5)
