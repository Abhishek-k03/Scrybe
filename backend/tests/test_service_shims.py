"""The `services/` layer after it became a shim over `app.rag`.

The routes were left alone, so these pin the contract they still depend on.
"""

from __future__ import annotations

import pytest

from app.rag.config import (
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    NoopRerankConfig,
    PipelineConfig,
)
from app.rag.types import Chunk, Document
from app.services import pipeline as app_pipeline
from app.services import store
from app.services.chunker import chunk_document, chunk_text
from app.services.retriever import as_dict, retrieve

TEXTS = [
    "Python uses reference counting to free memory automatically",
    "The global interpreter lock prevents true thread parallelism",
    "Django is a high level web framework that ships with an ORM",
]


@pytest.fixture
def offline(monkeypatch):
    """Repoint the app's config at the offline embedder, keeping the temp index."""
    real_config = app_pipeline.default_config

    def config() -> PipelineConfig:
        return PipelineConfig(
            chunk=FixedCharChunkConfig(chunk_size=120, overlap=20),
            embed=FakeEmbedConfig(dimensions=128),
            index=real_config().index,
            retrieve=DenseRetrieveConfig(top_k=5),
            rerank=NoopRerankConfig(),
        )

    monkeypatch.setattr(app_pipeline, "default_config", config)
    app_pipeline.reset()
    yield
    app_pipeline.reset()


def make_chunks(doc_id: str = "doc1", texts: list[str] = TEXTS) -> list[Chunk]:
    return [
        Chunk(
            doc_id=doc_id,
            doc_label=f"{doc_id}.txt",
            source_type="file",
            chunk_index=i,
            text=text,
            start_char=i * 100,
            end_char=i * 100 + len(text),
        )
        for i, text in enumerate(texts)
    ]


def vectors_for(chunks: list[Chunk]) -> list[list[float]]:
    from app.rag.embed.fake import embed_texts

    return embed_texts([c.text for c in chunks], FakeEmbedConfig(dimensions=128))


# --------------------------------------------------------------------------------------
# singletons
# --------------------------------------------------------------------------------------


def test_the_index_is_built_from_settings(_isolated_chroma) -> None:
    from app.core.config import settings

    assert app_pipeline.default_config().index.path == settings.CHROMA_PATH


def test_the_app_index_is_writable() -> None:
    """read_only defaults to True in the config; the server has to opt out."""
    assert app_pipeline.default_config().index.read_only is False


def test_production_defaults_are_unchanged() -> None:
    """Chunking and reranking change when an eval says so, not as refactor fallout."""
    config = app_pipeline.default_config()

    assert (config.chunk.kind, config.chunk.chunk_size, config.chunk.overlap) == (
        "fixed_char",
        800,
        150,
    )
    assert config.rerank.kind == "noop"
    assert config.retrieve.kind == "dense"


def test_the_config_carries_no_secret() -> None:
    assert "api_key" not in app_pipeline.default_config().to_json().lower()


def test_the_server_caches_query_embeddings() -> None:
    """Rule 3 applied to production: a repeated question must not be re-embedded."""
    assert app_pipeline.default_config().embed.cache_dir is not None


def test_the_embedding_cache_can_be_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty setting means no cache, not a directory literally named "" in the CWD."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBED_CACHE_PATH", "", raising=False)
    assert app_pipeline.default_config().embed.cache_dir is None


def test_the_cache_path_comes_from_settings(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from app.core.config import settings

    monkeypatch.setattr(settings, "EMBED_CACHE_PATH", str(tmp_path), raising=False)
    assert app_pipeline.default_config().embed.cache_dir == str(tmp_path)


def test_the_pipeline_and_the_store_share_one_index(_isolated_chroma) -> None:
    assert app_pipeline.get().index is store.get_index()


def test_reset_rebuilds_against_the_current_settings(_isolated_chroma) -> None:
    first = store.get_index()
    app_pipeline.reset()
    assert store.get_index() is not first


# --------------------------------------------------------------------------------------
# chunker
# --------------------------------------------------------------------------------------


def test_chunk_text_still_returns_only_text_and_index() -> None:
    for chunk in chunk_text("word " * 400):
        assert set(chunk) == {"text", "chunk_index"}


def test_chunk_document_keeps_the_offsets_chunk_text_throws_away() -> None:
    doc = Document.create("d.txt", "word " * 400)
    for chunk in chunk_document(doc):
        assert doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_chunk_document_uses_the_configured_size(offline) -> None:
    doc = Document.create("d.txt", "word " * 400)
    assert all(len(chunk.text) <= 120 for chunk in chunk_document(doc))


def test_chunk_document_and_chunk_text_agree_on_the_same_parameters(offline) -> None:
    text = "word " * 400
    doc = Document.create("d.txt", text)

    assert [c.text for c in chunk_document(doc)] == [
        c["text"] for c in chunk_text(text, chunk_size=120, overlap=20)
    ]


# --------------------------------------------------------------------------------------
# store
# --------------------------------------------------------------------------------------


def test_add_chunks_stores_them(_isolated_chroma) -> None:
    chunks = make_chunks()
    assert store.add_chunks(chunks, vectors_for(chunks)) == 3
    assert store.count_total_chunks() == 3


def test_add_chunks_of_nothing_is_zero(_isolated_chroma) -> None:
    assert store.add_chunks([], []) == 0


def test_adding_the_same_document_twice_stores_nothing(_isolated_chroma) -> None:
    chunks = make_chunks()
    store.add_chunks(chunks, vectors_for(chunks))

    assert store.add_chunks(chunks, vectors_for(chunks)) == 0
    assert store.count_total_chunks() == 3


def test_sources_are_grouped_by_document(_isolated_chroma) -> None:
    first = make_chunks("aaa", TEXTS[:2])
    second = make_chunks("bbb", TEXTS[2:])
    store.add_chunks(first, vectors_for(first))
    store.add_chunks(second, vectors_for(second))

    listing = {entry["source_id"]: entry["chunk_count"] for entry in store.get_all_sources()}
    assert listing == {"aaa": 2, "bbb": 1}


def test_source_listing_keeps_the_label_and_type(_isolated_chroma) -> None:
    chunks = make_chunks()
    store.add_chunks(chunks, vectors_for(chunks))

    entry = store.get_all_sources()[0]
    assert (entry["source_label"], entry["source_type"]) == ("doc1.txt", "file")


def test_delete_removes_only_the_named_document(_isolated_chroma) -> None:
    first = make_chunks("aaa", TEXTS[:2])
    second = make_chunks("bbb", TEXTS[2:])
    store.add_chunks(first, vectors_for(first))
    store.add_chunks(second, vectors_for(second))

    assert store.delete_source("aaa") == 2
    assert store.count_total_chunks() == 1


def test_deleting_an_absent_document_is_a_noop(_isolated_chroma) -> None:
    assert store.delete_source("missing") == 0


def test_deleting_frees_the_id_for_reingest(_isolated_chroma) -> None:
    chunks = make_chunks()
    store.add_chunks(chunks, vectors_for(chunks))
    store.delete_source("doc1")

    assert store.add_chunks(chunks, vectors_for(chunks)) == 3


# --------------------------------------------------------------------------------------
# retriever
# --------------------------------------------------------------------------------------


async def test_retrieve_returns_the_keys_the_routes_read(offline, _isolated_chroma) -> None:
    chunks = make_chunks()
    store.add_chunks(chunks, vectors_for(chunks))

    hits = await retrieve("global interpreter lock", top_k=2)

    assert len(hits) == 2
    for hit in hits:
        assert {"text", "source_id", "source_label", "source_type", "chunk_index"} <= set(hit)


def test_distance_is_a_distance_not_a_similarity() -> None:
    """Chroma's value is lower-is-better; `score` is the flipped one."""
    from app.rag.types import Hit

    hit = Hit(chunk=make_chunks()[0], score=0.8, distance=0.2)
    flattened = as_dict(hit)

    assert flattened["distance"] == 0.2
    assert flattened["score"] == 0.8


def test_distance_is_derived_when_the_backend_did_not_supply_one() -> None:
    from app.rag.types import Hit

    assert as_dict(Hit(chunk=make_chunks()[0], score=0.75))["distance"] == 0.25


async def test_retrieve_on_a_blank_question_returns_nothing(offline, _isolated_chroma) -> None:
    chunks = make_chunks()
    store.add_chunks(chunks, vectors_for(chunks))

    assert await retrieve("   ") == []


async def test_retrieve_ranks_the_lexically_closest_chunk_first(
    offline, _isolated_chroma
) -> None:
    chunks = make_chunks()
    store.add_chunks(chunks, vectors_for(chunks))

    hits = await retrieve("django web framework orm", top_k=3)
    assert "Django" in hits[0]["text"]


async def test_retrieval_sees_a_document_added_after_the_first_query(
    offline, _isolated_chroma
) -> None:
    first = make_chunks("aaa", TEXTS[:2])
    store.add_chunks(first, vectors_for(first))
    await retrieve("django", top_k=2)

    second = make_chunks("bbb", TEXTS[2:])
    store.add_chunks(second, vectors_for(second))

    assert any(hit["source_id"] == "bbb" for hit in await retrieve("django", top_k=3))
