"""The abstention curve.

The sweep is exercised end to end against a mocked reranker, because the thing worth
guarding is that only the floor moves between points — same index, same scores, same
queries — so the shape of the curve is attributable to the threshold and nothing else.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from app.rag.config import (
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    JinaRerankConfig,
    MemoryIndexConfig,
    NoopRerankConfig,
    PipelineConfig,
)
from app.rag.pipeline import build_pipeline
from app.rag.rerank import jina
from app.rag.types import Document
from evals import abstention
from evals.label_schema import GoldSpan, LabelledQuery, LabelSet
from evals.metrics import QueryOutcome

DOCS = [
    Document.create(
        "gil.txt",
        "The global interpreter lock means only one thread may execute bytecode at a time.",
    ),
    Document.create(
        "guido.txt",
        "Guido van Rossum created Python in 1989 and released it in 1991.",
    ),
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
        LabelledQuery(id="q2", query="what is the airspeed of a swallow", answerable=False),
    ],
)


def reranked_config(**overrides) -> PipelineConfig:
    base = {
        "chunk": FixedCharChunkConfig(chunk_size=200, overlap=40),
        "embed": FakeEmbedConfig(dimensions=128),
        "index": MemoryIndexConfig(),
        "retrieve": DenseRetrieveConfig(top_k=5, fetch_k=10),
        "rerank": JinaRerankConfig(),
    }
    return PipelineConfig(**{**base, **overrides})


def outcome(query_id: str, *, answerable: bool, retrieved: tuple[str, ...]) -> QueryOutcome:
    return QueryOutcome(
        query_id=query_id,
        answerable=answerable,
        retrieved_chunk_ids=retrieved,
        retrieved_doc_ids=tuple(cid.split("-")[0] for cid in retrieved),
        relevant_chunk_ids=frozenset({"d-0"}) if answerable else frozenset(),
        relevant_doc_ids=frozenset({"d"}) if answerable else frozenset(),
        retrieved_chunk_chars=tuple(100 for _ in retrieved),
    )


# --------------------------------------------------------------------------------------
# applying the floor to a config
# --------------------------------------------------------------------------------------


def test_the_threshold_lands_on_the_rerank_stage() -> None:
    assert abstention.with_threshold(reranked_config(), 0.25).rerank.score_threshold == 0.25


def test_none_clears_the_threshold() -> None:
    config = reranked_config(rerank=JinaRerankConfig(score_threshold=0.9))
    assert abstention.with_threshold(config, None).rerank.score_threshold is None


def test_nothing_else_in_the_config_moves() -> None:
    """If the chunker or the pool moved too, the curve would not isolate the threshold."""
    config = reranked_config()
    shifted = abstention.with_threshold(config, 0.2)

    assert shifted.chunk == config.chunk
    assert shifted.retrieve == config.retrieve
    assert shifted.embed == config.embed


def test_a_stage_with_no_score_is_refused() -> None:
    """noop leaves the retriever's distance in place; a floor on it means something else."""
    with pytest.raises(abstention.NotReRankedError, match="noop"):
        abstention.with_threshold(reranked_config(rerank=NoopRerankConfig()), 0.2)


# --------------------------------------------------------------------------------------
# one point on the curve
# --------------------------------------------------------------------------------------


def test_abstaining_on_an_unanswerable_query_is_the_win() -> None:
    row = abstention.curve_row(
        0.2,
        [
            outcome("q1", answerable=True, retrieved=("d-0",)),
            outcome("q2", answerable=False, retrieved=()),
        ],
        5,
    )
    assert row["abstention_rate"] == 1.0
    assert row["false_abstention_rate"] == 0.0


def test_abstaining_on_an_answerable_query_is_the_cost() -> None:
    row = abstention.curve_row(
        0.9,
        [
            outcome("q1", answerable=True, retrieved=()),
            outcome("q2", answerable=False, retrieved=()),
        ],
        5,
    )
    assert row["false_abstention_rate"] == 1.0
    assert row["recall@5"] == 0.0


def test_the_row_names_who_abstained() -> None:
    """A rate of 0.4 over 5 queries is not actionable; knowing which two were is."""
    row = abstention.curve_row(
        0.2,
        [
            outcome("q1", answerable=True, retrieved=("d-0",)),
            outcome("q2", answerable=False, retrieved=()),
        ],
        5,
    )
    assert row["abstained_query_ids"] == ["q2"]


def test_the_row_carries_its_own_threshold() -> None:
    row = abstention.curve_row(0.15, [outcome("q1", answerable=True, retrieved=("d-0",))], 5)
    assert row["score_threshold"] == 0.15


# --------------------------------------------------------------------------------------
# the sweep
# --------------------------------------------------------------------------------------


@pytest.fixture
def mocked_reranker(monkeypatch: pytest.MonkeyPatch):
    """Score every candidate at 0.2, so a floor above it abstains and below it does not."""
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": i, "relevance_score": 0.2}
                    for i in range(len(payload["documents"]))
                ]
            },
        )

    real_client = httpx.AsyncClient
    monkeypatch.setattr(
        jina.httpx,
        "AsyncClient",
        lambda *a, **kw: real_client(*a, **{**kw, "transport": httpx.MockTransport(handler)}),
    )
    return calls


async def test_the_curve_has_a_point_per_threshold_plus_the_unfiltered_baseline(
    mocked_reranker,
) -> None:
    artifact = await abstention.sweep(
        reranked_config(), LABELS, DOCS, "key", (0.1, 0.3), k=5
    )
    assert [row["score_threshold"] for row in artifact["curve"]] == [None, 0.1, 0.3]


async def test_a_floor_below_the_scores_changes_nothing(mocked_reranker) -> None:
    artifact = await abstention.sweep(reranked_config(), LABELS, DOCS, "key", (0.1,), k=5)
    baseline, floored = artifact["curve"]
    assert floored["recall@5"] == baseline["recall@5"]
    assert floored["abstention_rate"] == 0.0


async def test_a_floor_above_the_scores_abstains_on_everything(mocked_reranker) -> None:
    artifact = await abstention.sweep(reranked_config(), LABELS, DOCS, "key", (0.9,), k=5)
    floored = artifact["curve"][-1]
    assert floored["abstention_rate"] == 1.0
    assert floored["false_abstention_rate"] == 1.0


async def test_the_reranker_is_called_once_per_query_not_once_per_threshold(
    mocked_reranker, tmp_path: Path
) -> None:
    """Rule 3: the scores do not depend on the floor, so the sweep must be free after the
    first pass. Without a cache the eight-point curve would cost eight times the calls."""
    config = reranked_config(rerank=JinaRerankConfig(cache_dir=str(tmp_path / "rerank")))
    await abstention.sweep(config, LABELS, DOCS, "key", (0.1, 0.2, 0.3, 0.4), k=5)

    assert len(mocked_reranker) == len(LABELS.queries)


async def test_the_index_is_built_once(mocked_reranker) -> None:
    """Every point after the first shares the index, so the corpus must not grow under it."""
    config = reranked_config()
    expected = sum(len(build_pipeline(config).chunk(doc)) for doc in DOCS)

    artifact = await abstention.sweep(config, LABELS, DOCS, "key", (0.1, 0.2), k=5)

    assert artifact["run"]["n_chunks"] == expected
    assert artifact["run"]["n_docs"] == 2


async def test_the_artifact_records_the_config_without_a_threshold(mocked_reranker) -> None:
    """The floor belongs to the curve rows; a leftover on the config would misread as global."""
    artifact = await abstention.sweep(reranked_config(), LABELS, DOCS, "key", (0.3,), k=5)
    assert artifact["config"]["rerank"]["score_threshold"] is None


async def test_the_artifact_records_the_label_author(mocked_reranker) -> None:
    artifact = await abstention.sweep(reranked_config(), LABELS, DOCS, "key", (0.3,), k=5)
    assert artifact["labels"]["author"] == "a test"


async def test_the_artifact_records_the_grid_it_swept(mocked_reranker) -> None:
    artifact = await abstention.sweep(reranked_config(), LABELS, DOCS, "key", (0.1, 0.3), k=5)
    assert artifact["run"]["thresholds"] == [0.1, 0.3]


# --------------------------------------------------------------------------------------
# the printed table
# --------------------------------------------------------------------------------------


async def test_the_table_says_how_small_the_sample_is(mocked_reranker) -> None:
    """A threshold tuned against five queries is fitted, and the output has to say so."""
    artifact = await abstention.sweep(reranked_config(), LABELS, DOCS, "key", (0.1,), k=5)
    text = abstention.format_curve(artifact)

    assert "1 unanswerable" in text
    assert "fitted to it" in text


async def test_the_unfiltered_row_is_labelled_rather_than_shown_as_zero(
    mocked_reranker,
) -> None:
    artifact = await abstention.sweep(reranked_config(), LABELS, DOCS, "key", (0.1,), k=5)
    assert "none" in abstention.format_curve(artifact)


def test_the_default_grid_is_ordered_and_declared() -> None:
    """Reordering points between runs would make two curves look different when they match."""
    assert list(abstention.THRESHOLDS) == sorted(abstention.THRESHOLDS)
