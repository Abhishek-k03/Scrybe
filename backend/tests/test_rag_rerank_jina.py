"""The Jina cross-encoder rerank stage.

Every HTTP call is served by a `MockTransport` except the one test marked `live`, which is
the only thing that can prove the request shape and response field names are still right.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from app.rag import registry
from app.rag.cache import DiskCache
from app.rag.config import JinaRerankConfig, PipelineConfig
from app.rag.rerank import jina
from app.rag.rerank.jina import (
    RerankResponseError,
    cache_key,
    order_by_score,
    score_documents,
    scores_from_response,
)
from app.rag.types import Chunk, Hit, RetrievalResult

DOCS = [
    "NumPy adds support for large multidimensional arrays.",
    "The global interpreter lock allows only one thread to execute bytecode at a time.",
    "Django is a free and open-source Python web framework.",
]


def make_hit(index: int, text: str, score: float = 0.0) -> Hit:
    return Hit(
        chunk=Chunk(
            doc_id="doc1",
            doc_label="doc1.txt",
            source_type="fixture",
            chunk_index=index,
            text=text,
            start_char=index * 100,
            end_char=index * 100 + len(text),
        ),
        score=score,
        distance=1.0 - score,
    )


def make_result(texts: Sequence[str] = DOCS, query: str = "what is the GIL") -> RetrievalResult:
    hits = tuple(make_hit(i, text, 0.9 - i / 10) for i, text in enumerate(texts))
    return RetrievalResult(query=query, hits=hits)


class Recorder:
    """A `MockTransport` handler that scores documents by position and logs each request."""

    def __init__(self, scores: dict[int, float] | None = None, status: int = 200) -> None:
        self.requests: list[dict] = []
        self.scores = scores
        self.status = status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        self.requests.append({"payload": payload, "headers": dict(request.headers)})
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "nope"})
        documents = payload["documents"]
        results = [
            {
                "index": i,
                "relevance_score": self.scores[i] if self.scores else -float(i),
            }
            for i in range(len(documents))
        ]
        return httpx.Response(200, json={"results": results})


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch):
    """Install a handler factory; the test picks the handler, the fixture wires it in."""

    def install(handler) -> None:
        real_client = httpx.AsyncClient

        def client(*args, **kwargs):
            return real_client(*args, **{**kwargs, "transport": httpx.MockTransport(handler)})

        monkeypatch.setattr(jina.httpx, "AsyncClient", client)

    return install


# --------------------------------------------------------------------------------------
# response parsing
# --------------------------------------------------------------------------------------


def test_scores_come_back_in_the_order_the_documents_were_sent() -> None:
    """The API returns results ranked, not aligned. Re-aligning is the whole job here."""
    payload = {
        "results": [
            {"index": 2, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
            {"index": 1, "relevance_score": 0.5},
        ]
    }
    assert scores_from_response(payload, 3) == [0.1, 0.5, 0.9]


def test_a_short_response_is_rejected() -> None:
    payload = {"results": [{"index": 0, "relevance_score": 0.9}]}
    with pytest.raises(RerankResponseError, match="expected 3 scores, got 1"):
        scores_from_response(payload, 3)


def test_an_out_of_range_index_is_rejected() -> None:
    payload = {"results": [{"index": 7, "relevance_score": 0.9}]}
    with pytest.raises(RerankResponseError, match="out of range or repeated"):
        scores_from_response(payload, 1)


def test_a_repeated_index_is_rejected() -> None:
    """Two scores for one document would leave another silently at zero."""
    payload = {
        "results": [
            {"index": 0, "relevance_score": 0.9},
            {"index": 0, "relevance_score": 0.1},
        ]
    }
    with pytest.raises(RerankResponseError, match="out of range or repeated"):
        scores_from_response(payload, 2)


# --------------------------------------------------------------------------------------
# ordering
# --------------------------------------------------------------------------------------


def test_hits_are_reordered_by_relevance() -> None:
    hits = [make_hit(i, text) for i, text in enumerate(DOCS)]
    ordered = order_by_score(hits, [0.1, 0.9, 0.5])
    assert [h.chunk.chunk_index for h in ordered] == [1, 2, 0]


def test_the_relevance_score_replaces_the_retriever_score() -> None:
    ordered = order_by_score([make_hit(0, DOCS[0], score=0.42)], [0.87])
    assert ordered[0].score == 0.87


def test_the_retriever_distance_survives_reranking() -> None:
    """Two different models produced these numbers; conflating them ruins the artifact."""
    hit = make_hit(0, DOCS[0], score=0.42)
    assert order_by_score([hit], [0.87])[0].distance == hit.distance


def test_ties_keep_the_retriever_order() -> None:
    hits = [make_hit(i, text) for i, text in enumerate(DOCS)]
    ordered = order_by_score(hits, [0.5, 0.5, 0.5])
    assert [h.chunk.chunk_index for h in ordered] == [0, 1, 2]


def test_negative_scores_sort_correctly() -> None:
    """jina-reranker-v3 returns unbounded logits, not values in [0, 1]."""
    hits = [make_hit(i, text) for i, text in enumerate(DOCS)]
    ordered = order_by_score(hits, [-0.9, 0.2, -0.1])
    assert [h.chunk.chunk_index for h in ordered] == [1, 2, 0]


# --------------------------------------------------------------------------------------
# request shape
# --------------------------------------------------------------------------------------


async def test_the_query_text_reaches_the_api(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    await score_documents("why is the GIL there", DOCS, JinaRerankConfig(), "key")
    assert recorder.requests[0]["payload"]["query"] == "why is the GIL there"


async def test_the_model_and_documents_are_sent_verbatim(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    config = JinaRerankConfig(model="jina-reranker-v2-base-multilingual")
    await score_documents("q", DOCS, config, "key")
    payload = recorder.requests[0]["payload"]
    assert payload["model"] == "jina-reranker-v2-base-multilingual"
    assert payload["documents"] == DOCS


async def test_documents_are_not_echoed_back(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    await score_documents("q", DOCS, JinaRerankConfig(), "key")
    assert recorder.requests[0]["payload"]["return_documents"] is False


async def test_the_key_travels_in_the_header(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    await score_documents("q", DOCS, JinaRerankConfig(), "sekrit")
    assert recorder.requests[0]["headers"]["authorization"] == "Bearer sekrit"


async def test_no_key_raises_before_any_request(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    with pytest.raises(RuntimeError, match="JINA_API_KEY is not configured"):
        await score_documents("q", DOCS, JinaRerankConfig(), "")
    assert recorder.requests == []


async def test_no_documents_does_not_need_a_key() -> None:
    assert await score_documents("q", [], JinaRerankConfig(), "") == []


async def test_a_blank_query_scores_nothing_rather_than_calling_out(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    assert await score_documents("   ", DOCS, JinaRerankConfig(), "key") == [0.0, 0.0, 0.0]
    assert recorder.requests == []


# --------------------------------------------------------------------------------------
# retries
# --------------------------------------------------------------------------------------


async def test_a_rate_limit_is_retried(transport, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jina.asyncio, "sleep", lambda _: _noop())
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(429, json={"detail": "slow down"})
        documents = json.loads(request.content)["documents"]
        return httpx.Response(
            200,
            json={"results": [{"index": i, "relevance_score": 0.0} for i in range(len(documents))]},
        )

    transport(handler)
    scores = await score_documents("q", DOCS, JinaRerankConfig(max_retries=3), "key")
    assert len(attempts) == 3 and len(scores) == 3


async def test_retries_are_bounded(transport, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(jina.asyncio, "sleep", lambda _: _noop())
    recorder = Recorder(status=503)
    transport(recorder)

    with pytest.raises(httpx.HTTPStatusError):
        await score_documents("q", DOCS, JinaRerankConfig(max_retries=2), "key")
    assert len(recorder.requests) == 3, "one initial attempt plus max_retries"


async def test_a_client_error_is_not_retried(transport) -> None:
    recorder = Recorder(status=401)
    transport(recorder)

    with pytest.raises(httpx.HTTPStatusError):
        await score_documents("q", DOCS, JinaRerankConfig(max_retries=3), "key")
    assert len(recorder.requests) == 1, "a bad key will not fix itself"


async def _noop() -> None:
    return None


# --------------------------------------------------------------------------------------
# caching
# --------------------------------------------------------------------------------------


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(tmp_path / "rerank")


async def test_the_second_identical_call_is_served_from_disk(transport, cache: DiskCache) -> None:
    recorder = Recorder()
    transport(recorder)
    config = JinaRerankConfig()

    first = await score_documents("q", DOCS, config, "key", cache=cache)
    second = await score_documents("q", DOCS, config, "key", cache=cache)

    assert first == second
    assert len(recorder.requests) == 1


async def test_a_different_query_is_a_different_entry(transport, cache: DiskCache) -> None:
    recorder = Recorder()
    transport(recorder)
    await score_documents("one", DOCS, JinaRerankConfig(), "key", cache=cache)
    await score_documents("two", DOCS, JinaRerankConfig(), "key", cache=cache)
    assert len(recorder.requests) == 2


async def test_a_different_candidate_set_is_a_different_entry(transport, cache: DiskCache) -> None:
    """Widening fetch_k must not read back the scores computed for the narrower pool."""
    recorder = Recorder()
    transport(recorder)
    await score_documents("q", DOCS[:2], JinaRerankConfig(), "key", cache=cache)
    await score_documents("q", DOCS, JinaRerankConfig(), "key", cache=cache)
    assert len(recorder.requests) == 2


async def test_a_different_model_is_a_different_entry(transport, cache: DiskCache) -> None:
    recorder = Recorder()
    transport(recorder)
    await score_documents("q", DOCS, JinaRerankConfig(model="a"), "key", cache=cache)
    await score_documents("q", DOCS, JinaRerankConfig(model="b"), "key", cache=cache)
    assert len(recorder.requests) == 2


def test_the_cache_key_is_over_texts_not_ids() -> None:
    """Chunk ids repeat across chunk sizes over different text; texts do not."""
    assert cache_key("m", "q", ["a", "b"]) != cache_key("m", "q", ["a", "c"])
    assert cache_key("m", "q", ["a", "b"]) == cache_key("m", "q", ["a", "b"])


async def test_an_entry_of_the_wrong_length_is_refetched(transport, cache: DiskCache) -> None:
    """A stored value that cannot align to the candidates is worse than no value."""
    recorder = Recorder()
    transport(recorder)
    config = JinaRerankConfig()
    cache.set(cache_key(config.model, "q", DOCS), [0.1])

    scores = await score_documents("q", DOCS, config, "key", cache=cache)
    assert len(scores) == 3 and len(recorder.requests) == 1


async def test_a_run_without_a_cache_dir_calls_out_every_time(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    await score_documents("q", DOCS, JinaRerankConfig(), "key")
    await score_documents("q", DOCS, JinaRerankConfig(), "key")
    assert len(recorder.requests) == 2


async def test_the_cache_survives_a_new_reranker(transport, tmp_path: Path) -> None:
    recorder = Recorder()
    transport(recorder)
    config = JinaRerankConfig(cache_dir=str(tmp_path / "rerank"))

    await registry.build("rerank", config, api_key="key")(make_result())
    await registry.build("rerank", config, api_key="key")(make_result())

    assert len(recorder.requests) == 1


# --------------------------------------------------------------------------------------
# the stage as built by the registry
# --------------------------------------------------------------------------------------


async def test_the_registered_stage_reorders_hits(transport) -> None:
    transport(Recorder(scores={0: 0.1, 1: 0.9, 2: 0.5}))
    reranker = registry.build("rerank", JinaRerankConfig(), api_key="key")
    ordered = await reranker(make_result())
    assert [h.chunk.chunk_index for h in ordered] == [1, 2, 0]


async def test_the_stage_truncates_to_top_k(transport) -> None:
    transport(Recorder())
    reranker = registry.build("rerank", JinaRerankConfig(top_k=2), api_key="key")
    assert len(await reranker(make_result())) == 2


async def test_the_stage_drops_hits_below_the_threshold(transport) -> None:
    transport(Recorder(scores={0: 0.9, 1: 0.4, 2: -0.2}))
    reranker = registry.build("rerank", JinaRerankConfig(score_threshold=0.5), api_key="key")
    ordered = await reranker(make_result())
    assert [h.chunk.chunk_index for h in ordered] == [0]


async def test_the_threshold_can_reject_everything(transport) -> None:
    """Retrieving nothing is a valid answer; it is what abstention is built on."""
    transport(Recorder(scores={0: 0.1, 1: 0.1, 2: 0.1}))
    reranker = registry.build("rerank", JinaRerankConfig(score_threshold=0.9), api_key="key")
    assert await reranker(make_result()) == []


async def test_the_stage_on_no_candidates_makes_no_request(transport) -> None:
    recorder = Recorder()
    transport(recorder)
    reranker = registry.build("rerank", JinaRerankConfig(), api_key="key")
    assert await reranker(RetrievalResult(query="q")) == []
    assert recorder.requests == []


def test_the_rerank_config_never_carries_a_secret() -> None:
    payload = json.dumps(PipelineConfig(rerank=JinaRerankConfig()).model_dump(mode="json")).lower()
    for banned in ("api_key", "apikey", "secret", "token=", "password"):
        assert banned not in payload


# --------------------------------------------------------------------------------------
# the real endpoint
# --------------------------------------------------------------------------------------


@pytest.mark.live
async def test_the_real_api_still_has_the_shape_we_parse() -> None:
    """Mocks pin our assumption about the contract; only this can falsify it."""
    import os

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=True)
    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        pytest.skip("JINA_API_KEY is not set")

    scores = await score_documents("what is the global interpreter lock", DOCS, JinaRerankConfig(), api_key)

    assert len(scores) == 3
    assert scores[1] == max(scores), f"the GIL passage should win; got {scores}"
