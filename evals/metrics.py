"""Retrieval metrics.

Pure functions over id lists, so they can be checked against hand-worked examples without
a corpus, an index, or a network.

Relevance is binary here: a chunk either contains a gold span or it does not. Metrics that
assume graded relevance (nDCG in its general form) are specialised to that case.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass


class UndefinedMetricError(ValueError):
    """Raised when a metric has no meaning for the query it was asked about.

    Recall over an empty relevant set is the usual case: an unanswerable query has no
    correct answer to have found, and scoring it 0 would silently drag the mean down.
    """


def recall_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Share of the relevant chunks that appear in the top k.

    Ceiling-limited when a query has more relevant chunks than k: 8 relevant chunks at
    k=5 cannot score above 0.625, which is a property of the labels, not the retriever.
    """
    if not relevant:
        raise UndefinedMetricError("recall is undefined when nothing is relevant")
    return len(set(retrieved[:k]) & set(relevant)) / len(relevant)


def precision_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    top = retrieved[:k]
    if not top:
        return 0.0
    return len(set(top) & set(relevant)) / len(top)


def hit_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """1.0 if anything relevant made the top k. The weakest useful signal."""
    return 1.0 if set(retrieved[:k]) & set(relevant) else 0.0


def reciprocal_rank(retrieved: Sequence[str], relevant: Collection[str]) -> float:
    """1/rank of the first relevant hit, or 0 if there is none."""
    for position, identifier in enumerate(retrieved, start=1):
        if identifier in relevant:
            return 1.0 / position
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(rank + 2) for rank, gain in enumerate(gains))


def ndcg_at_k(retrieved: Sequence[str], relevant: Collection[str], k: int) -> float:
    """Discounted cumulative gain against the best possible ordering.

    The ideal ranking puts min(len(relevant), k) relevant chunks first, so a query with
    more relevant chunks than k can still reach 1.0 — unlike recall@k.
    """
    if not relevant:
        raise UndefinedMetricError("nDCG is undefined when nothing is relevant")
    gains = [1.0 if identifier in relevant else 0.0 for identifier in retrieved[:k]]
    ideal = dcg([1.0] * min(len(relevant), k))
    return dcg(gains) / ideal if ideal else 0.0


@dataclass(frozen=True)
class QueryOutcome:
    """What retrieval returned for one labelled query."""

    query_id: str
    answerable: bool
    retrieved_chunk_ids: tuple[str, ...]
    retrieved_doc_ids: tuple[str, ...]
    relevant_chunk_ids: frozenset[str]
    relevant_doc_ids: frozenset[str]
    # Length of each retrieved chunk, in rank order. Bigger chunks raise recall@k for free
    # unless the context they cost is reported next to it.
    retrieved_chunk_chars: tuple[int, ...] = ()

    @property
    def abstained(self) -> bool:
        return not self.retrieved_chunk_ids

    def chars_at_k(self, k: int) -> int:
        """Characters a caller would hand to the LLM if it took the top k."""
        return sum(self.retrieved_chunk_chars[:k])


def _mean(values: Sequence[float]) -> float | None:
    """None rather than 0.0 for an empty sample, so a missing measurement cannot read
    as a bad one."""
    return sum(values) / len(values) if values else None


def summarise(outcomes: Sequence[QueryOutcome], k_values: Sequence[int]) -> dict[str, object]:
    """Aggregate per-query outcomes into the numbers that go in a result artifact.

    Answerable and unanswerable queries are kept apart throughout: ranking quality is only
    defined for the first group, abstention only for the second.
    """
    answerable = [outcome for outcome in outcomes if outcome.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.answerable]

    # An answerable query whose gold span straddles a chunk boundary has no relevant chunk
    # to find under this chunking config. Scoring it 0 would blame the retriever for a
    # labelling artifact; dropping it quietly would move the denominator between configs
    # and make a chunk-size sweep incomparable. So it is excluded and counted out loud.
    scorable = [outcome for outcome in answerable if outcome.relevant_chunk_ids]
    unreachable = [outcome for outcome in answerable if not outcome.relevant_chunk_ids]

    summary: dict[str, object] = {
        "n_queries": len(outcomes),
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "n_scored": len(scorable),
        "n_unreachable": len(unreachable),
        "unreachable_query_ids": [outcome.query_id for outcome in unreachable],
    }

    for k in k_values:
        summary[f"recall@{k}"] = _mean(
            [recall_at_k(o.retrieved_chunk_ids, o.relevant_chunk_ids, k) for o in scorable]
        )
        summary[f"doc_recall@{k}"] = _mean(
            [recall_at_k(o.retrieved_doc_ids, o.relevant_doc_ids, k) for o in scorable]
        )
        summary[f"ndcg@{k}"] = _mean(
            [ndcg_at_k(o.retrieved_chunk_ids, o.relevant_chunk_ids, k) for o in scorable]
        )
        summary[f"hit@{k}"] = _mean(
            [hit_at_k(o.retrieved_chunk_ids, o.relevant_chunk_ids, k) for o in scorable]
        )
        # The context budget that recall@k was bought with, so a chunk-size comparison
        # is not read as free improvement.
        summary[f"chars@{k}"] = _mean([float(o.chars_at_k(k)) for o in scorable])

    summary["mrr"] = _mean(
        [reciprocal_rank(o.retrieved_chunk_ids, o.relevant_chunk_ids) for o in scorable]
    )
    # Retrieval correctly returning nothing when the corpus cannot answer.
    summary["abstention_rate"] = _mean([1.0 if o.abstained else 0.0 for o in unanswerable])
    # Retrieval returning nothing when it should have found something.
    summary["false_abstention_rate"] = _mean([1.0 if o.abstained else 0.0 for o in scorable])

    return summary
