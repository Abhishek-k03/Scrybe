"""Embedder behavior, caching, and the guarantees that make sweeps cheap and honest."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from app.rag import registry
from app.rag.cache import DiskCache, content_key
from app.rag.config import FakeEmbedConfig, JinaEmbedConfig, PipelineConfig
from app.rag.embed.cached import cached_embedder
from app.rag.embed.fake import embed_one, embed_texts
from app.rag.embed.jina import task_for
from app.rag.protocols import EmbedKind


class SpyEmbedder:
    """Records every batch it is asked to embed."""

    def __init__(self, dimensions: int = 4) -> None:
        self.calls: list[list[str]] = []
        self.dimensions = dimensions

    async def __call__(
        self, texts: Sequence[str], kind: EmbedKind = "passage"
    ) -> list[list[float]]:
        self.calls.append(list(texts))
        return [[float(len(text))] * self.dimensions for text in texts]

    @property
    def texts_embedded(self) -> int:
        return sum(len(batch) for batch in self.calls)


@pytest.fixture
def cache(tmp_path: Path) -> DiskCache:
    return DiskCache(tmp_path / "embeddings")


def passage_only(kind: EmbedKind) -> str:
    return "retrieval.query" if kind == "query" else "retrieval.passage"


# --------------------------------------------------------------------------------------
# content keys
# --------------------------------------------------------------------------------------


def test_key_is_stable_for_the_same_parts() -> None:
    assert content_key("a", "b") == content_key("a", "b")


def test_key_separates_parts_unambiguously() -> None:
    """('ab','c') must not collide with ('a','bc')."""
    assert content_key("ab", "c") != content_key("a", "bc")


def test_key_changes_with_model_and_with_task() -> None:
    base = content_key("jina-v3", "retrieval.passage", "hello")
    assert base != content_key("jina-v4", "retrieval.passage", "hello")
    assert base != content_key("jina-v3", "retrieval.query", "hello")


# --------------------------------------------------------------------------------------
# disk cache
# --------------------------------------------------------------------------------------


def test_cache_round_trips_a_vector(cache: DiskCache) -> None:
    cache.set("abc123", [0.1, 0.2])
    assert cache.get("abc123") == [0.1, 0.2]


def test_cache_miss_returns_none(cache: DiskCache) -> None:
    assert cache.get("nothing") is None


def test_cache_counts_hits_and_misses(cache: DiskCache) -> None:
    cache.get("absent")
    cache.set("present", [1.0])
    cache.get("present")
    assert (cache.hits, cache.misses) == (1, 1)


def test_cache_survives_a_truncated_file(cache: DiskCache) -> None:
    """An interrupted write must read as a miss, not crash the run."""
    cache.set("key", [1.0])
    cache.path_for("key").write_text("{not json", encoding="utf-8")
    assert cache.get("key") is None


def test_cache_writes_are_atomic(cache: DiskCache) -> None:
    cache.set("key", [1.0, 2.0])
    leftovers = list(cache.root.rglob("*.tmp"))
    assert not leftovers, f"temp files left behind: {leftovers}"


def test_cache_persists_across_instances(tmp_path: Path) -> None:
    DiskCache(tmp_path).set("k", [3.0])
    assert DiskCache(tmp_path).get("k") == [3.0]


# --------------------------------------------------------------------------------------
# cached wrapper
# --------------------------------------------------------------------------------------


async def test_second_call_does_not_reach_the_inner_embedder(cache: DiskCache) -> None:
    spy = SpyEmbedder()
    embedder = cached_embedder(spy, cache, model="m", task_for=passage_only)

    first = await embedder(["alpha", "beta"])
    second = await embedder(["alpha", "beta"])

    assert first == second
    assert spy.texts_embedded == 2, "the second call should have been served from disk"


async def test_only_missing_texts_are_sent_to_the_inner_embedder(cache: DiskCache) -> None:
    spy = SpyEmbedder()
    embedder = cached_embedder(spy, cache, model="m", task_for=passage_only)

    await embedder(["alpha"])
    await embedder(["alpha", "beta", "gamma"])

    assert spy.calls == [["alpha"], ["beta", "gamma"]]


async def test_results_keep_input_order_when_partially_cached(cache: DiskCache) -> None:
    """An off-by-one here would attach the wrong vector to every chunk."""
    spy = SpyEmbedder()
    embedder = cached_embedder(spy, cache, model="m", task_for=passage_only)

    await embedder(["bb"])
    result = await embedder(["a", "bb", "ccc", "dddd"])

    assert [vector[0] for vector in result] == [1.0, 2.0, 3.0, 4.0]


async def test_duplicate_texts_in_one_batch_cost_one_call(cache: DiskCache) -> None:
    spy = SpyEmbedder()
    embedder = cached_embedder(spy, cache, model="m", task_for=passage_only)

    result = await embedder(["same", "other", "same"])

    assert spy.calls == [["same", "other"]]
    assert result[0] == result[2]


async def test_query_and_passage_do_not_share_cache_entries(cache: DiskCache) -> None:
    spy = SpyEmbedder()
    embedder = cached_embedder(spy, cache, model="m", task_for=passage_only)

    await embedder(["text"], "passage")
    await embedder(["text"], "query")

    assert spy.texts_embedded == 2, "query and passage embeddings must not collide"


async def test_changing_the_model_invalidates_the_cache(cache: DiskCache) -> None:
    spy = SpyEmbedder()
    await cached_embedder(spy, cache, model="v3", task_for=passage_only)(["text"])
    await cached_embedder(spy, cache, model="v4", task_for=passage_only)(["text"])

    assert spy.texts_embedded == 2


async def test_empty_input_never_calls_the_inner_embedder(cache: DiskCache) -> None:
    spy = SpyEmbedder()
    assert await cached_embedder(spy, cache, model="m", task_for=passage_only)([]) == []
    assert spy.calls == []


async def test_a_short_inner_response_is_rejected(cache: DiskCache) -> None:
    class ShortEmbedder:
        async def __call__(self, texts, kind="passage"):
            return [[1.0]]

    embedder = cached_embedder(ShortEmbedder(), cache, model="m", task_for=passage_only)
    with pytest.raises(RuntimeError, match="1 vectors for 2 texts"):
        await embedder(["a", "b"])


# --------------------------------------------------------------------------------------
# fake embedder
# --------------------------------------------------------------------------------------


def test_fake_embedding_is_deterministic() -> None:
    config = FakeEmbedConfig()
    assert embed_one("hello world", config) == embed_one("hello world", config)


def test_fake_embedding_has_the_configured_width() -> None:
    assert len(embed_one("text", FakeEmbedConfig(dimensions=32))) == 32


def test_fake_embedding_is_unit_length() -> None:
    vector = embed_one("some words here", FakeEmbedConfig())
    assert abs(sum(v * v for v in vector) - 1.0) < 1e-9


def test_fake_embedding_of_empty_text_is_still_a_unit_vector() -> None:
    vector = embed_one("", FakeEmbedConfig())
    assert abs(sum(v * v for v in vector) - 1.0) < 1e-9


def test_fake_embedding_preserves_lexical_similarity() -> None:
    """Shared words must score closer than unrelated text, or ranking tests are vacuous."""
    config = FakeEmbedConfig(dimensions=256)
    query = embed_one("python garbage collection reference counting", config)
    related = embed_one("reference counting is how python frees memory", config)
    unrelated = embed_one("counter strike weapon skins tournament", config)

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b, strict=True))

    assert dot(query, related) > dot(query, unrelated)


def test_fake_embedding_changes_with_the_seed() -> None:
    assert embed_one("text", FakeEmbedConfig(seed=1)) != embed_one("text", FakeEmbedConfig(seed=2))


def test_fake_embed_texts_matches_embed_one() -> None:
    config = FakeEmbedConfig()
    texts = ["one", "two", "three"]
    assert embed_texts(texts, config) == [embed_one(t, config) for t in texts]


async def test_registry_builds_a_working_fake_embedder() -> None:
    embedder = registry.build("embed", FakeEmbedConfig(dimensions=16))
    vectors = await embedder(["a", "b"])
    assert len(vectors) == 2 and len(vectors[0]) == 16


# --------------------------------------------------------------------------------------
# jina config plumbing
# --------------------------------------------------------------------------------------


def test_task_string_depends_on_the_kind() -> None:
    config = JinaEmbedConfig()
    assert task_for(config, "passage") == "retrieval.passage"
    assert task_for(config, "query") == "retrieval.query"


async def test_jina_without_a_key_raises_before_any_request() -> None:
    embedder = registry.build("embed", JinaEmbedConfig())
    with pytest.raises(RuntimeError, match="JINA_API_KEY is not configured"):
        await embedder(["text"])


async def test_jina_with_no_texts_does_not_need_a_key() -> None:
    assert await registry.build("embed", JinaEmbedConfig())([]) == []


def test_config_never_carries_a_secret() -> None:
    """Config is serialised into eval artifacts, so keys must travel separately."""
    payload = json.dumps(PipelineConfig(embed=JinaEmbedConfig()).model_dump(mode="json")).lower()
    for banned in ("api_key", "apikey", "secret", "token=", "password"):
        assert banned not in payload
