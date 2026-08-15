"""Retrieval metrics, against hand-worked numbers.

A metric that is subtly wrong produces plausible results forever, so the expected values
here are derived by hand rather than recorded from the implementation.
"""

from __future__ import annotations

import math

import pytest

from evals.metrics import (
    QueryOutcome,
    UndefinedMetricError,
    dcg,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    summarise,
)

RANKING = ["a", "b", "c", "d", "e"]


def outcome(
    query_id: str = "q1",
    *,
    answerable: bool = True,
    retrieved: tuple[str, ...] = ("a", "b"),
    relevant: frozenset[str] = frozenset({"a"}),
    retrieved_docs: tuple[str, ...] = ("d1", "d1"),
    relevant_docs: frozenset[str] = frozenset({"d1"}),
    chars: tuple[int, ...] | None = None,
) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        answerable=answerable,
        retrieved_chunk_ids=retrieved,
        retrieved_doc_ids=retrieved_docs,
        relevant_chunk_ids=relevant,
        relevant_doc_ids=relevant_docs,
        retrieved_chunk_chars=chars if chars is not None else (100,) * len(retrieved),
    )


# --------------------------------------------------------------------------------------
# recall
# --------------------------------------------------------------------------------------


def test_recall_counts_only_the_top_k() -> None:
    assert recall_at_k(RANKING, {"a", "e"}, k=3) == 0.5


def test_recall_of_a_perfect_ranking_is_one() -> None:
    assert recall_at_k(RANKING, {"a", "b"}, k=2) == 1.0


def test_recall_of_nothing_found_is_zero() -> None:
    assert recall_at_k(RANKING, {"z"}, k=5) == 0.0


def test_recall_is_capped_when_relevant_exceeds_k() -> None:
    """8 relevant chunks at k=5 cannot beat 0.625 — a property of the labels."""
    relevant = {f"r{i}" for i in range(8)}
    retrieved = [f"r{i}" for i in range(5)]
    assert recall_at_k(retrieved, relevant, k=5) == 0.625


def test_recall_over_an_empty_relevant_set_raises() -> None:
    """Scoring an unanswerable query 0 would silently drag the mean down."""
    with pytest.raises(UndefinedMetricError, match="undefined"):
        recall_at_k(RANKING, set(), k=5)


def test_duplicate_retrievals_do_not_inflate_recall() -> None:
    assert recall_at_k(["a", "a", "a"], {"a", "b"}, k=3) == 0.5


# --------------------------------------------------------------------------------------
# precision and hit rate
# --------------------------------------------------------------------------------------


def test_precision_is_over_what_was_returned() -> None:
    assert precision_at_k(RANKING, {"a", "b"}, k=4) == 0.5


def test_precision_of_an_empty_result_is_zero_not_an_error() -> None:
    assert precision_at_k([], {"a"}, k=5) == 0.0


def test_hit_is_binary() -> None:
    assert hit_at_k(RANKING, {"e"}, k=5) == 1.0
    assert hit_at_k(RANKING, {"e"}, k=4) == 0.0


# --------------------------------------------------------------------------------------
# reciprocal rank
# --------------------------------------------------------------------------------------


def test_reciprocal_rank_of_the_first_position_is_one() -> None:
    assert reciprocal_rank(RANKING, {"a"}) == 1.0


def test_reciprocal_rank_uses_the_first_relevant_hit() -> None:
    assert reciprocal_rank(RANKING, {"c", "d"}) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_was_found() -> None:
    assert reciprocal_rank(RANKING, {"z"}) == 0.0


def test_reciprocal_rank_of_an_empty_ranking_is_zero() -> None:
    assert reciprocal_rank([], {"a"}) == 0.0


# --------------------------------------------------------------------------------------
# nDCG
# --------------------------------------------------------------------------------------


def test_dcg_discounts_by_log2_of_rank_plus_one() -> None:
    assert dcg([1.0, 1.0]) == pytest.approx(1.0 + 1 / math.log2(3))


def test_ndcg_of_a_perfect_ranking_is_one() -> None:
    assert ndcg_at_k(RANKING, {"a", "b"}, k=5) == pytest.approx(1.0)


def test_ndcg_penalises_a_lower_rank() -> None:
    """One relevant hit at rank 2 scores 1/log2(3) against an ideal of 1."""
    assert ndcg_at_k(RANKING, {"b"}, k=3) == pytest.approx(1 / math.log2(3))


def test_ndcg_reaches_one_even_when_relevant_exceeds_k() -> None:
    """Unlike recall, the ideal ranking is truncated to k, so a full top-k scores 1.0."""
    relevant = {"a", "b", "c", "d", "e", "f"}
    assert ndcg_at_k(RANKING, relevant, k=2) == pytest.approx(1.0)
    assert recall_at_k(RANKING, relevant, k=2) == pytest.approx(2 / 6)


def test_ndcg_of_nothing_relevant_found_is_zero() -> None:
    assert ndcg_at_k(RANKING, {"z"}, k=5) == 0.0


def test_ndcg_over_an_empty_relevant_set_raises() -> None:
    with pytest.raises(UndefinedMetricError):
        ndcg_at_k(RANKING, set(), k=5)


def test_ndcg_is_order_sensitive_where_recall_is_not() -> None:
    front = ndcg_at_k(["a", "z", "y"], {"a"}, k=3)
    back = ndcg_at_k(["z", "y", "a"], {"a"}, k=3)

    assert front > back
    assert recall_at_k(["a", "z", "y"], {"a"}, 3) == recall_at_k(["z", "y", "a"], {"a"}, 3)


# --------------------------------------------------------------------------------------
# aggregation
# --------------------------------------------------------------------------------------


def test_sample_sizes_travel_with_the_numbers() -> None:
    """No metric should be readable without knowing how many queries produced it."""
    summary = summarise([outcome(), outcome("q2", answerable=False, relevant=frozenset())], [5])

    assert summary["n_queries"] == 2
    assert summary["n_answerable"] == 1
    assert summary["n_unanswerable"] == 1


def test_unanswerable_queries_are_excluded_from_ranking_metrics() -> None:
    """Including them would score a correct abstention as a retrieval failure."""
    only_answerable = summarise([outcome()], [1])
    with_unanswerable = summarise(
        [outcome(), outcome("q2", answerable=False, retrieved=(), relevant=frozenset())], [1]
    )
    assert only_answerable["recall@1"] == with_unanswerable["recall@1"] == 1.0


def test_an_empty_group_reports_none_not_zero() -> None:
    """A missing measurement must not read as a bad one."""
    summary = summarise([outcome()], [1])
    assert summary["abstention_rate"] is None
    assert summary["recall@1"] == 1.0


def test_abstention_is_measured_over_unanswerable_queries_only() -> None:
    outcomes = [
        outcome("u1", answerable=False, retrieved=(), relevant=frozenset()),
        outcome("u2", answerable=False, retrieved=("a",), relevant=frozenset()),
        outcome("q1"),
    ]
    assert summarise(outcomes, [1])["abstention_rate"] == 0.5


def test_a_retriever_that_never_abstains_scores_zero() -> None:
    """The behaviour the audit flagged: without a threshold this is always 0."""
    outcomes = [outcome(f"u{i}", answerable=False, retrieved=("a",), relevant=frozenset()) for i in range(3)]
    assert summarise(outcomes, [1])["abstention_rate"] == 0.0


def test_false_abstention_catches_a_threshold_set_too_high() -> None:
    outcomes = [outcome("q1", retrieved=()), outcome("q2")]
    assert summarise(outcomes, [1])["false_abstention_rate"] == 0.5


def test_document_recall_is_reported_beside_chunk_recall() -> None:
    """Several chunks of one document can satisfy a query at document level but not chunk."""
    single = outcome(
        retrieved=("c1", "c2"),
        relevant=frozenset({"c1", "c9"}),
        retrieved_docs=("d1", "d1"),
        relevant_docs=frozenset({"d1"}),
    )
    summary = summarise([single], [2])

    assert summary["recall@2"] == 0.5
    assert summary["doc_recall@2"] == 1.0


def test_every_requested_k_appears_in_the_summary() -> None:
    summary = summarise([outcome()], [1, 3, 5])
    for k in (1, 3, 5):
        for name in ("recall", "doc_recall", "ndcg", "hit"):
            assert f"{name}@{k}" in summary


def test_summarising_no_queries_at_all_does_not_crash() -> None:
    summary = summarise([], [5])
    assert summary["n_queries"] == 0
    assert summary["mrr"] is None


# --------------------------------------------------------------------------------------
# unreachable queries — the trap in a chunk-size sweep
# --------------------------------------------------------------------------------------


def unreachable(query_id: str = "x1") -> QueryOutcome:
    """Answerable, but its gold span sits across a chunk boundary at this chunk size."""
    return outcome(query_id, relevant=frozenset(), relevant_docs=frozenset())


def test_an_unreachable_query_is_excluded_rather_than_scored_zero() -> None:
    """Scoring it 0 would blame the retriever for a labelling artifact."""
    summary = summarise([outcome("q1"), unreachable()], [1])

    assert summary["recall@1"] == 1.0
    assert summary["n_scored"] == 1
    assert summary["n_unreachable"] == 1


def test_unreachable_queries_are_named_not_just_counted() -> None:
    """Silent exclusion moves the denominator between runs with nothing to show for it."""
    summary = summarise([outcome("q1"), unreachable("q8"), unreachable("q9")], [1])
    assert summary["unreachable_query_ids"] == ["q8", "q9"]


def test_the_answerable_count_still_includes_unreachable_queries() -> None:
    """n_answerable and n_scored differing is the signal that a comparison is unsafe."""
    summary = summarise([outcome("q1"), unreachable()], [1])

    assert summary["n_answerable"] == 2
    assert summary["n_scored"] == 1


def test_a_fully_unreachable_label_set_scores_nothing_rather_than_zero() -> None:
    summary = summarise([unreachable("a"), unreachable("b")], [5])

    assert summary["recall@5"] is None
    assert summary["mrr"] is None
    assert summary["n_unreachable"] == 2


# --------------------------------------------------------------------------------------
# context budget
# --------------------------------------------------------------------------------------


def test_chars_at_k_sums_only_the_top_k() -> None:
    single = outcome(retrieved=("a", "b", "c"), chars=(100, 200, 400))
    assert single.chars_at_k(2) == 300
    assert single.chars_at_k(99) == 700


def test_the_context_budget_is_reported_next_to_recall() -> None:
    """Doubling chunk size raises recall@k; without this it looks free."""
    small = summarise([outcome(chars=(400, 400))], [2])
    large = summarise([outcome(chars=(1600, 1600))], [2])

    assert small["recall@2"] == large["recall@2"] == 1.0
    assert small["chars@2"] == 800.0
    assert large["chars@2"] == 3200.0


def test_the_budget_is_reported_at_every_k() -> None:
    summary = summarise([outcome(retrieved=("a", "b"), chars=(100, 200))], [1, 3])
    assert summary["chars@1"] == 100.0
    assert summary["chars@3"] == 300.0
