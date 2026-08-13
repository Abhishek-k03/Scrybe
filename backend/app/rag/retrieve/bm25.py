"""Okapi BM25 lexical scoring.

Used as one arm of hybrid retrieval. Dense embeddings miss exact identifiers — a query for
a specific function or version string often has no semantic neighbourhood to land in.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass

from app.rag.types import Chunk

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass(frozen=True)
class Bm25Index:
    chunks: tuple[Chunk, ...]
    term_frequencies: tuple[Counter[str], ...]
    lengths: tuple[int, ...]
    average_length: float
    document_frequency: dict[str, int]

    @property
    def size(self) -> int:
        return len(self.chunks)


def build_bm25(chunks: Sequence[Chunk]) -> Bm25Index:
    frequencies = tuple(Counter(tokenize(chunk.text)) for chunk in chunks)
    lengths = tuple(sum(counter.values()) for counter in frequencies)
    document_frequency: dict[str, int] = {}
    for counter in frequencies:
        for term in counter:
            document_frequency[term] = document_frequency.get(term, 0) + 1

    average = (sum(lengths) / len(lengths)) if lengths else 0.0
    return Bm25Index(
        chunks=tuple(chunks),
        term_frequencies=frequencies,
        lengths=lengths,
        average_length=average,
        document_frequency=document_frequency,
    )


def score_query(index: Bm25Index, query: str, k1: float, b: float) -> list[float]:
    terms = tokenize(query)
    scores = [0.0] * index.size
    if not terms or index.size == 0 or index.average_length == 0.0:
        return scores

    total = index.size
    for term in terms:
        df = index.document_frequency.get(term, 0)
        if df == 0:
            continue
        idf = math.log(1.0 + (total - df + 0.5) / (df + 0.5))
        for position, counter in enumerate(index.term_frequencies):
            frequency = counter.get(term, 0)
            if not frequency:
                continue
            norm = 1.0 - b + b * (index.lengths[position] / index.average_length)
            scores[position] += idf * (frequency * (k1 + 1.0)) / (frequency + k1 * norm)
    return scores


def rank(index: Bm25Index, query: str, top_k: int, k1: float, b: float) -> list[tuple[int, float]]:
    """Positions and scores of the best matches, highest first, zero-scoring dropped."""
    scores = score_query(index, query, k1, b)
    ordered = sorted(
        (pair for pair in enumerate(scores) if pair[1] > 0.0),
        key=lambda pair: (-pair[1], pair[0]),
    )
    return ordered[:top_k]
