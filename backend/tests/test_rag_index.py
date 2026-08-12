"""Index behavior, the read-only guard, and memory/Chroma agreement."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag import registry
from app.rag.config import ChromaIndexConfig, FakeEmbedConfig, MemoryIndexConfig
from app.rag.embed.fake import embed_texts
from app.rag.index.chroma import ChromaIndex, ReadOnlyIndexError
from app.rag.index.memory import MemoryIndex
from app.rag.types import Chunk

EMBED = FakeEmbedConfig(dimensions=64)

TEXTS = [
    "Python uses reference counting to free memory automatically",
    "The global interpreter lock prevents true thread parallelism",
    "NumPy provides fast multidimensional array operations",
    "Django is a high level web framework with an ORM",
    "Guido van Rossum created the Python language in 1991",
]


def make_chunks(texts: list[str] = TEXTS, doc_id: str = "doc1") -> list[Chunk]:
    return [
        Chunk(
            doc_id=doc_id,
            doc_label=f"{doc_id}.txt",
            source_type="fixture",
            chunk_index=i,
            text=text,
            start_char=i * 100,
            end_char=i * 100 + len(text),
        )
        for i, text in enumerate(texts)
    ]


def vectors_for(chunks: list[Chunk]) -> list[list[float]]:
    return embed_texts([c.text for c in chunks], EMBED)


@pytest.fixture
def memory_index() -> MemoryIndex:
    index = MemoryIndex(MemoryIndexConfig())
    chunks = make_chunks()
    index.add(chunks, vectors_for(chunks))
    return index


@pytest.fixture
def chroma_index(tmp_path: Path) -> ChromaIndex:
    index = ChromaIndex(ChromaIndexConfig(path=str(tmp_path / "idx"), read_only=False))
    chunks = make_chunks()
    index.add(chunks, vectors_for(chunks))
    return index


# --------------------------------------------------------------------------------------
# memory index
# --------------------------------------------------------------------------------------


def test_add_returns_the_number_stored(memory_index: MemoryIndex) -> None:
    assert memory_index.count() == len(TEXTS)


def test_search_returns_top_k_in_score_order(memory_index: MemoryIndex) -> None:
    query = embed_texts(["how does python free memory"], EMBED)[0]
    hits = memory_index.search(query, top_k=3)

    assert len(hits) == 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_search_finds_the_lexically_closest_chunk(memory_index: MemoryIndex) -> None:
    query = embed_texts(["global interpreter lock thread parallelism"], EMBED)[0]
    assert "global interpreter lock" in memory_index.search(query, top_k=1)[0].chunk.text


def test_score_and_distance_are_consistent(memory_index: MemoryIndex) -> None:
    query = embed_texts(["numpy arrays"], EMBED)[0]
    for hit in memory_index.search(query, top_k=3):
        assert abs(hit.score - (1.0 - hit.distance)) < 1e-12


def test_search_is_deterministic(memory_index: MemoryIndex) -> None:
    query = embed_texts(["python"], EMBED)[0]
    first = memory_index.search(query, top_k=5)
    for _ in range(5):
        assert memory_index.search(query, top_k=5) == first


def test_ties_keep_insertion_order() -> None:
    """Equal scores must not reorder between runs."""
    index = MemoryIndex(MemoryIndexConfig())
    chunks = make_chunks(["same text", "same text", "same text"])
    index.add(chunks, [[1.0, 0.0]] * 3)

    hits = index.search([1.0, 0.0], top_k=3)
    assert [h.chunk.chunk_index for h in hits] == [0, 1, 2]


def test_top_k_larger_than_the_index_returns_everything(memory_index: MemoryIndex) -> None:
    query = embed_texts(["python"], EMBED)[0]
    assert len(memory_index.search(query, top_k=99)) == len(TEXTS)


def test_search_on_an_empty_index_returns_nothing() -> None:
    assert MemoryIndex(MemoryIndexConfig()).search([1.0, 0.0], top_k=5) == []


def test_zero_top_k_returns_nothing(memory_index: MemoryIndex) -> None:
    query = embed_texts(["python"], EMBED)[0]
    assert memory_index.search(query, top_k=0) == []


def test_mismatched_chunk_and_embedding_counts_are_rejected() -> None:
    index = MemoryIndex(MemoryIndexConfig())
    with pytest.raises(ValueError, match="2 chunks but 1 embeddings"):
        index.add(make_chunks(["a", "b"]), [[1.0]])


def test_mismatched_embedding_width_is_rejected(memory_index: MemoryIndex) -> None:
    with pytest.raises(ValueError, match="does not match index width"):
        memory_index.add(make_chunks(["x"]), [[1.0, 2.0]])


def test_mismatched_query_width_is_rejected(memory_index: MemoryIndex) -> None:
    with pytest.raises(ValueError, match="query width"):
        memory_index.search([1.0, 2.0], top_k=1)


def test_delete_removes_only_the_named_document() -> None:
    index = MemoryIndex(MemoryIndexConfig())
    keep = make_chunks(["alpha"], doc_id="keep")
    drop = make_chunks(["beta", "gamma"], doc_id="drop")
    index.add(keep + drop, vectors_for(keep + drop))

    assert index.delete_doc("drop") == 2
    assert index.count() == 1
    assert index.search(vectors_for(keep)[0], top_k=5)[0].chunk.doc_id == "keep"


def test_delete_of_an_absent_document_is_a_noop(memory_index: MemoryIndex) -> None:
    assert memory_index.delete_doc("nope") == 0
    assert memory_index.count() == len(TEXTS)


# --------------------------------------------------------------------------------------
# read-only guard
# --------------------------------------------------------------------------------------


def test_writing_to_a_read_only_index_raises(tmp_path: Path) -> None:
    path = str(tmp_path / "idx")
    ChromaIndex(ChromaIndexConfig(path=path, read_only=False))

    index = ChromaIndex(ChromaIndexConfig(path=path))
    chunks = make_chunks(["x"])
    with pytest.raises(ReadOnlyIndexError, match="read-only"):
        index.add(chunks, vectors_for(chunks))


def test_deleting_from_a_read_only_index_raises(tmp_path: Path) -> None:
    path = str(tmp_path / "idx")
    ChromaIndex(ChromaIndexConfig(path=path, read_only=False))

    with pytest.raises(ReadOnlyIndexError, match="read-only"):
        ChromaIndex(ChromaIndexConfig(path=path)).delete_doc("doc1")


def test_read_only_open_of_a_missing_collection_fails(tmp_path: Path) -> None:
    """Read-only must not conjure an empty collection that silently scores zero."""
    with pytest.raises(Exception, match="(?i)does not exist|not found"):
        ChromaIndex(ChromaIndexConfig(path=str(tmp_path / "empty")))


def test_read_only_index_can_still_search(chroma_index: ChromaIndex) -> None:
    reader = ChromaIndex(ChromaIndexConfig(path=chroma_index.config.path))
    query = embed_texts(["numpy arrays"], EMBED)[0]
    assert len(reader.search(query, top_k=3)) == 3


# --------------------------------------------------------------------------------------
# coexistence
# --------------------------------------------------------------------------------------


def test_two_indexes_coexist_in_one_process(tmp_path: Path) -> None:
    """The old module-global client made this impossible."""
    first = ChromaIndex(ChromaIndexConfig(path=str(tmp_path / "a"), read_only=False))
    second = ChromaIndex(ChromaIndexConfig(path=str(tmp_path / "b"), read_only=False))

    a_chunks = make_chunks(["alpha only"], doc_id="a")
    b_chunks = make_chunks(["beta only", "gamma too"], doc_id="b")
    first.add(a_chunks, vectors_for(a_chunks))
    second.add(b_chunks, vectors_for(b_chunks))

    assert (first.count(), second.count()) == (1, 2)


def test_same_path_different_collections_stay_separate(tmp_path: Path) -> None:
    path = str(tmp_path / "shared")
    one = ChromaIndex(ChromaIndexConfig(path=path, collection="one", read_only=False))
    two = ChromaIndex(ChromaIndexConfig(path=path, collection="two", read_only=False))

    chunks = make_chunks(["only in one"], doc_id="x")
    one.add(chunks, vectors_for(chunks))

    assert (one.count(), two.count()) == (1, 0)


# --------------------------------------------------------------------------------------
# backend agreement
# --------------------------------------------------------------------------------------


def test_memory_and_chroma_rank_identically(
    memory_index: MemoryIndex, chroma_index: ChromaIndex
) -> None:
    """Guards against silent ANN drift once the corpus grows past exact search."""
    for question in ["how does python free memory", "web framework orm", "who made python"]:
        query = embed_texts([question], EMBED)[0]
        mem = [h.chunk.text for h in memory_index.search(query, top_k=5)]
        chroma = [h.chunk.text for h in chroma_index.search(query, top_k=5)]
        assert mem == chroma, f"backends disagree for {question!r}"


def test_memory_and_chroma_agree_on_scores(
    memory_index: MemoryIndex, chroma_index: ChromaIndex
) -> None:
    query = embed_texts(["global interpreter lock"], EMBED)[0]
    mem = memory_index.search(query, top_k=5)
    chroma = chroma_index.search(query, top_k=5)

    for a, b in zip(mem, chroma, strict=True):
        assert abs(a.score - b.score) < 1e-5


def test_chroma_round_trips_chunk_metadata(chroma_index: ChromaIndex) -> None:
    """Offsets must survive storage, or span-keyed labels break at retrieval time."""
    query = embed_texts([TEXTS[2]], EMBED)[0]
    hit = chroma_index.search(query, top_k=1)[0]
    original = make_chunks()[2]

    assert hit.chunk == original


# --------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------


def test_both_index_kinds_are_registered() -> None:
    assert registry.registered_kinds("index") == ("chroma", "memory")


def test_registry_builds_a_memory_index() -> None:
    assert isinstance(registry.build("index", MemoryIndexConfig()), MemoryIndex)


def test_registry_builds_a_chroma_index(tmp_path: Path) -> None:
    config = ChromaIndexConfig(path=str(tmp_path / "r"), read_only=False)
    assert isinstance(registry.build("index", config), ChromaIndex)
