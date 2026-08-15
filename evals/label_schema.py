"""Schema and relevance rule for the hand-authored retrieval labels.

Labels are keyed to (document, gold span), never to chunk ids. A chunk id encodes the
chunk_size that produced it, so the first chunking sweep would silently repoint every label
at different text — the file would still parse, the metrics would still compute, and they
would be wrong. Instead:

    a chunk is relevant to a query when it comes from the labelled document
    and contains the gold span verbatim

That definition survives re-chunking, which is what makes a chunk-size sweep comparable.

Read-only by construction: this module loads and checks labels, and nothing in the repo
writes to evals/labels/.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.types import Chunk

# A span long enough to straddle a chunk boundary is unmatchable at small chunk sizes.
LONG_SPAN_CHARS = 300


class GoldSpan(BaseModel):
    """One passage that answers a query, quoted verbatim from a corpus document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    doc: str = Field(min_length=1, description="corpus filename, e.g. 'cpython.txt'")
    span: str = Field(min_length=1, description="verbatim substring of that document")


class LabelledQuery(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    answerable: bool = True
    gold: tuple[GoldSpan, ...] = ()
    notes: str = ""

    @model_validator(mode="after")
    def _gold_matches_answerability(self) -> LabelledQuery:
        # Stated explicitly rather than inferred from an empty list, so a half-finished
        # label cannot pass as a deliberate unanswerable one.
        if self.answerable and not self.gold:
            raise ValueError(f"{self.id}: answerable query has no gold spans")
        if not self.answerable and self.gold:
            raise ValueError(f"{self.id}: unanswerable query must have no gold spans")
        return self


class LabelSet(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    version: Literal[1] = 1
    corpus_sha256: str = Field(min_length=64, max_length=64)
    # Required, so no label set can exist without saying where its judgements came from.
    # Every result artifact copies this: who decided what counts as a correct retrieval is
    # part of what a retrieval number means.
    author: str = Field(min_length=1)
    queries: tuple[LabelledQuery, ...]

    @model_validator(mode="after")
    def _ids_are_unique(self) -> LabelSet:
        seen = [query.id for query in self.queries]
        duplicates = sorted({name for name in seen if seen.count(name) > 1})
        if duplicates:
            raise ValueError(f"duplicate query ids: {duplicates}")
        return self

    @property
    def answerable(self) -> tuple[LabelledQuery, ...]:
        return tuple(query for query in self.queries if query.answerable)

    @property
    def unanswerable(self) -> tuple[LabelledQuery, ...]:
        return tuple(query for query in self.queries if not query.answerable)


def load(path: str | Path) -> LabelSet:
    return LabelSet.model_validate_json(Path(path).read_text(encoding="utf-8"))


def is_relevant(chunk: Chunk, query: LabelledQuery, doc_ids: Mapping[str, str]) -> bool:
    """Whether a retrieved chunk counts as a hit for this query."""
    return any(
        chunk.doc_id == doc_ids.get(gold.doc) and gold.span in chunk.text
        for gold in query.gold
    )


def relevant_chunks(
    chunks: Iterable[Chunk], query: LabelledQuery, doc_ids: Mapping[str, str]
) -> list[Chunk]:
    """Every chunk in the index that satisfies this query's labels.

    This is the denominator for recall, and it depends on the chunking config: overlap can
    put one span in two chunks, and a span that straddles a boundary lands in none.
    """
    return [chunk for chunk in chunks if is_relevant(chunk, query, doc_ids)]


def check(labels: LabelSet, documents: Sequence, corpus_sha256: str) -> list[str]:
    """Problems that would make these labels produce wrong numbers. Empty means usable.

    `documents` are the loaded corpus `Document`s; their `label` is the filename a gold
    span refers to.
    """
    problems: list[str] = []

    if labels.corpus_sha256 != corpus_sha256:
        # Full hash, not truncated: it is the value that has to be pasted into the file.
        problems.append(
            f"corpus hash mismatch: labels pin {labels.corpus_sha256[:12]}… but the corpus "
            f"is {corpus_sha256} — paste that in, or find out why the corpus moved"
        )

    by_label = {doc.label: doc for doc in documents}

    for query in labels.queries:
        for gold in query.gold:
            doc = by_label.get(gold.doc)
            if doc is None:
                problems.append(f"{query.id}: no corpus document named {gold.doc!r}")
                continue

            occurrences = doc.text.count(gold.span)
            if occurrences == 0:
                problems.append(
                    f"{query.id}: span not found verbatim in {gold.doc} — {gold.span[:60]!r}"
                )
            elif occurrences > 1:
                problems.append(
                    f"{query.id}: span appears {occurrences}x in {gold.doc}, so which "
                    f"passage is gold is ambiguous — {gold.span[:60]!r}"
                )
            if len(gold.span) > LONG_SPAN_CHARS:
                problems.append(
                    f"{query.id}: span is {len(gold.span)} chars; anything over "
                    f"{LONG_SPAN_CHARS} risks straddling a chunk boundary and matching "
                    f"nothing at small chunk sizes"
                )

    return problems


def warnings(labels: LabelSet) -> list[str]:
    """Things that weaken the eval without making it wrong."""
    notes: list[str] = []
    if "claude" in labels.author.lower() or "gpt" in labels.author.lower():
        notes.append(
            f"labels authored by {labels.author}: the model that built the retriever also "
            "decided what counts as a correct retrieval, so these numbers measure agreement "
            "with that model's judgement, not independent ground truth"
        )
    if not labels.unanswerable:
        notes.append(
            "no unanswerable queries: abstention cannot be measured, so a retriever that "
            "never says 'I don't know' will score the same as one that can"
        )
    single = [q.id for q in labels.answerable if len(q.gold) == 1]
    if len(single) > len(labels.answerable) * 0.8:
        notes.append(
            f"{len(single)}/{len(labels.answerable)} answerable queries have a single gold "
            "span; recall@k saturates quickly when every query has one right answer"
        )
    return notes


def corpus_hash(manifest_path: str | Path) -> str:
    return json.loads(Path(manifest_path).read_text(encoding="utf-8"))["corpus_sha256"]
