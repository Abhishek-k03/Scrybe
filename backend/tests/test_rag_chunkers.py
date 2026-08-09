"""Chunker behavior, offsets, and parity with the original implementation."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from app.rag import registry
from app.rag.chunk.fixed_char import chunk_fixed_char
from app.rag.chunk.sentence import chunk_sentences, sentence_spans
from app.rag.chunk.token import chunk_tokens
from app.rag.config import FixedCharChunkConfig, SentenceChunkConfig, TokenChunkConfig
from app.rag.types import Document
from app.services.chunker import chunk_text

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text() -> str:
    return (FIXTURES / "sample_doc.txt").read_text(encoding="utf-8")


@pytest.fixture
def sample_doc(sample_text: str) -> Document:
    return Document.create(label="sample_doc.txt", text=sample_text)


class WordTokenizer:
    """Deterministic offline stand-in for tiktoken. Round-trips exactly."""

    def __init__(self) -> None:
        self._vocab: list[str] = []
        self._ids: dict[str, int] = {}

    def encode(self, text: str) -> list[int]:
        pieces = re.findall(r"\s+|\S+", text)
        out = []
        for piece in pieces:
            if piece not in self._ids:
                self._ids[piece] = len(self._vocab)
                self._vocab.append(piece)
            out.append(self._ids[piece])
        return out

    def decode(self, tokens: list[int]) -> str:
        return "".join(self._vocab[t] for t in tokens)


# --------------------------------------------------------------------------------------
# parity with the original chunker
# --------------------------------------------------------------------------------------


def test_fixed_char_matches_the_original_on_the_fixture(sample_doc: Document) -> None:
    produced = chunk_fixed_char(sample_doc, FixedCharChunkConfig())
    expected = chunk_text(sample_doc.text)

    assert [c.text for c in produced] == [c["text"] for c in expected]
    assert [c.chunk_index for c in produced] == [c["chunk_index"] for c in expected]


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(800, 150), (500, 100), (100, 0), (64, 63), (1000, 1), (37, 11)],
)
def test_fixed_char_matches_the_original_across_parameters(
    sample_doc: Document, chunk_size: int, overlap: int
) -> None:
    config = FixedCharChunkConfig(chunk_size=chunk_size, overlap=overlap)
    produced = chunk_fixed_char(sample_doc, config)
    expected = chunk_text(sample_doc.text, chunk_size=chunk_size, overlap=overlap)

    assert [c.text for c in produced] == [c["text"] for c in expected]


CORPUS_DIR = Path(__file__).resolve().parents[2] / "evals" / "corpus"
CORPUS_FILES = sorted(CORPUS_DIR.glob("*.txt"))


@pytest.mark.skipif(not CORPUS_FILES, reason="corpus not fetched")
@pytest.mark.parametrize("path", CORPUS_FILES, ids=lambda p: p.stem)
def test_fixed_char_matches_the_original_across_the_whole_corpus(path: Path) -> None:
    """Parity on 35 real documents, not just the hand-written fixture."""
    text = path.read_text(encoding="utf-8")
    doc = Document.create(label=path.name, text=text)

    for chunk_size, overlap in [(800, 150), (400, 40), (1200, 300)]:
        config = FixedCharChunkConfig(chunk_size=chunk_size, overlap=overlap)
        produced = chunk_fixed_char(doc, config)
        expected = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        assert [c.text for c in produced] == [c["text"] for c in expected], (
            f"{path.stem} diverged at {chunk_size}/{overlap}"
        )
        for chunk in produced:
            assert text[chunk.start_char : chunk.end_char] == chunk.text


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   \n\n\t ",
        "short",
        "x" * 800,
        "x" * 801,
        "ab" + (" " * 10) + "cd",
        "para one.\n\n\n\npara two.",
    ],
    ids=["empty", "blank", "short", "exact", "over-by-one", "blank-window", "paragraphs"],
)
def test_fixed_char_matches_the_original_on_edge_cases(text: str) -> None:
    doc = Document.create(label="edge", text=text)
    config = FixedCharChunkConfig(chunk_size=5, overlap=1)
    produced = chunk_fixed_char(doc, config)
    expected = chunk_text(text, chunk_size=5, overlap=1)

    assert [c.text for c in produced] == [c["text"] for c in expected]


# --------------------------------------------------------------------------------------
# offsets
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("chunk_size,overlap", [(800, 150), (200, 50), (40, 5)])
def test_fixed_char_offsets_slice_back_to_the_chunk(
    sample_doc: Document, chunk_size: int, overlap: int
) -> None:
    config = FixedCharChunkConfig(chunk_size=chunk_size, overlap=overlap)
    for chunk in chunk_fixed_char(sample_doc, config):
        assert sample_doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_sentence_offsets_slice_back_to_the_chunk(sample_doc: Document) -> None:
    for chunk in chunk_sentences(sample_doc, SentenceChunkConfig(max_chars=400)):
        assert sample_doc.text[chunk.start_char : chunk.end_char] == chunk.text


def test_token_offsets_slice_back_to_the_chunk(sample_doc: Document) -> None:
    config = TokenChunkConfig(max_tokens=60, overlap_tokens=10)
    for chunk in chunk_tokens(sample_doc, config, WordTokenizer()):
        assert sample_doc.text[chunk.start_char : chunk.end_char] == chunk.text


@pytest.mark.parametrize("chunk_size,overlap", [(800, 150), (200, 50)])
def test_fixed_char_chunks_advance_monotonically(
    sample_doc: Document, chunk_size: int, overlap: int
) -> None:
    chunks = chunk_fixed_char(sample_doc, FixedCharChunkConfig(chunk_size=chunk_size, overlap=overlap))
    starts = [c.start_char for c in chunks]
    assert starts == sorted(starts)
    assert all(c.end_char > c.start_char for c in chunks)


def test_fixed_char_covers_every_non_space_character(sample_doc: Document) -> None:
    """No content may fall between two windows."""
    config = FixedCharChunkConfig(chunk_size=200, overlap=50)
    covered = set()
    for chunk in chunk_fixed_char(sample_doc, config):
        covered.update(range(chunk.start_char, chunk.end_char))

    missed = [
        i for i, ch in enumerate(sample_doc.text) if not ch.isspace() and i not in covered
    ]
    assert not missed, f"{len(missed)} characters fell between windows"


# --------------------------------------------------------------------------------------
# sentence chunker
# --------------------------------------------------------------------------------------


def test_sentence_spans_trim_whitespace() -> None:
    text = "One.  Two!  Three?"
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == ["One.", "Two!", "Three?"]


def test_sentence_spans_split_on_blank_lines() -> None:
    text = "First para\n\nSecond para"
    spans = sentence_spans(text)
    assert [text[a:b] for a, b in spans] == ["First para", "Second para"]


def test_sentence_chunks_do_not_split_mid_sentence() -> None:
    text = " ".join(f"Sentence number {i} here." for i in range(20))
    doc = Document.create(label="s", text=text)
    for chunk in chunk_sentences(doc, SentenceChunkConfig(max_chars=100)):
        assert chunk.text.endswith(".")


def test_sentence_chunk_respects_the_budget_when_it_can() -> None:
    text = " ".join(f"Sentence {i}." for i in range(40))
    doc = Document.create(label="s", text=text)
    chunks = chunk_sentences(doc, SentenceChunkConfig(max_chars=80, overlap_sentences=0))
    assert all(len(c.text) <= 80 for c in chunks)


def test_oversized_sentence_becomes_its_own_chunk() -> None:
    long_sentence = "word " * 100 + "end."
    doc = Document.create(label="s", text=f"Short one. {long_sentence} Short two.")
    chunks = chunk_sentences(doc, SentenceChunkConfig(max_chars=50, overlap_sentences=0))
    assert any(len(c.text) > 50 for c in chunks)


_SIX_SENTENCES = "Aa one. Bb two. Cc three. Dd four. Ee five. Ff six."


def test_sentence_overlap_repeats_the_previous_sentence() -> None:
    doc = Document.create(label="s", text=_SIX_SENTENCES)
    chunks = chunk_sentences(doc, SentenceChunkConfig(max_chars=25, overlap_sentences=1))

    assert len(chunks) > 1
    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert later.start_char < earlier.end_char


def test_no_overlap_when_overlap_sentences_is_zero() -> None:
    doc = Document.create(label="s", text=_SIX_SENTENCES)
    chunks = chunk_sentences(doc, SentenceChunkConfig(max_chars=25, overlap_sentences=0))

    for earlier, later in zip(chunks, chunks[1:], strict=False):
        assert later.start_char >= earlier.end_char


def test_carried_sentence_is_dropped_when_it_would_break_the_budget() -> None:
    """Budget wins over overlap, otherwise max_chars would not bound anything."""
    doc = Document.create(label="s", text="Alpha one. Bravo two. Charlie three. Delta four.")
    chunks = chunk_sentences(doc, SentenceChunkConfig(max_chars=25, overlap_sentences=1))

    assert [c.text for c in chunks] == [
        "Alpha one. Bravo two.",
        "Bravo two. Charlie three.",
        "Delta four.",
    ]


def test_sentence_chunker_on_empty_text() -> None:
    doc = Document.create(label="s", text="")
    assert chunk_sentences(doc, SentenceChunkConfig()) == []


# --------------------------------------------------------------------------------------
# token chunker
# --------------------------------------------------------------------------------------


def test_token_chunks_respect_the_token_budget(sample_doc: Document) -> None:
    tokenizer = WordTokenizer()
    config = TokenChunkConfig(max_tokens=50, overlap_tokens=10)
    for chunk in chunk_tokens(sample_doc, config, tokenizer):
        assert len(tokenizer.encode(chunk.text)) <= config.max_tokens


def test_token_chunker_covers_the_whole_document(sample_doc: Document) -> None:
    config = TokenChunkConfig(max_tokens=50, overlap_tokens=10)
    chunks = chunk_tokens(sample_doc, config, WordTokenizer())
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(sample_doc.text.rstrip())


def test_token_chunker_on_empty_text() -> None:
    doc = Document.create(label="t", text="")
    assert chunk_tokens(doc, TokenChunkConfig(), WordTokenizer()) == []


def test_token_chunker_single_chunk_when_under_budget() -> None:
    doc = Document.create(label="t", text="just a few words here")
    chunks = chunk_tokens(doc, TokenChunkConfig(max_tokens=500), WordTokenizer())
    assert len(chunks) == 1
    assert chunks[0].text == doc.text


# --------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------


def test_all_chunk_kinds_are_registered() -> None:
    assert registry.registered_kinds("chunk") == ("fixed_char", "sentence", "token")


def test_registry_builds_a_working_fixed_char_chunker(sample_doc: Document) -> None:
    chunker = registry.build("chunk", FixedCharChunkConfig())
    assert [c.text for c in chunker(sample_doc)] == [
        c["text"] for c in chunk_text(sample_doc.text)
    ]


def test_registry_builds_a_working_sentence_chunker(sample_doc: Document) -> None:
    chunker = registry.build("chunk", SentenceChunkConfig(max_chars=300))
    chunks = chunker(sample_doc)
    assert chunks and all(c.doc_id == sample_doc.doc_id for c in chunks)


def test_chunk_ids_are_unique_within_a_document(sample_doc: Document) -> None:
    chunks = chunk_fixed_char(sample_doc, FixedCharChunkConfig())
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
