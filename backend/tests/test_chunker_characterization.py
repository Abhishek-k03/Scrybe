"""Characterization tests for the current chunker.

These do not assert that `chunk_text` is *good*. They assert that it behaves exactly as it
behaves today, quirks included, so that the `fixed_char` chunker built in Phase 2 can be
proven byte-identical before anything downstream is allowed to change.

Read a failure here as "retrieval behavior changed", not "a test is out of date". The only
legitimate reason to update the golden file is a deliberate, measured decision recorded in
an evals/results/ artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.services.chunker import chunk_text

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = Path(__file__).parent / "golden"

DEFAULT_CHUNK_SIZE = 800
DEFAULT_OVERLAP = 150
DEFAULT_STRIDE = DEFAULT_CHUNK_SIZE - DEFAULT_OVERLAP  # 650


@pytest.fixture
def sample_doc() -> str:
    return (FIXTURES / "sample_doc.txt").read_text(encoding="utf-8")


# --------------------------------------------------------------------------------------
# Golden output — the Phase 2 parity oracle
# --------------------------------------------------------------------------------------


def test_golden_output_is_unchanged(sample_doc: str) -> None:
    """The full chunking of a realistic document, pinned exactly."""
    expected = json.loads((GOLDEN / "chunker_800_150.json").read_text(encoding="utf-8"))
    assert chunk_text(sample_doc) == expected


def test_golden_matches_the_documented_defaults(sample_doc: str) -> None:
    """Passing the defaults explicitly must equal calling with none.

    `ingest.py` calls `chunk_text(text)` with no arguments, so the signature defaults are
    the production configuration. This pins them at 800/150.
    """
    assert chunk_text(sample_doc) == chunk_text(sample_doc, chunk_size=800, overlap=150)


# --------------------------------------------------------------------------------------
# The windowing algorithm
# --------------------------------------------------------------------------------------


def test_windows_start_at_exact_stride_multiples(sample_doc: str) -> None:
    """Each chunk is `text[i : i+800].strip()` for i stepping by 650."""
    chunks = chunk_text(sample_doc)
    starts = [i * DEFAULT_STRIDE for i in range(len(chunks))]

    for chunk, start in zip(chunks, starts, strict=True):
        raw = sample_doc[start : start + DEFAULT_CHUNK_SIZE]
        assert chunk["text"] == raw.strip()


def test_final_window_covers_the_tail(sample_doc: str) -> None:
    """The loop breaks once a window reaches the end, so the tail is never dropped."""
    chunks = chunk_text(sample_doc)
    assert sample_doc.rstrip().endswith(chunks[-1]["text"][-40:])


def test_consecutive_raw_windows_share_the_overlap(sample_doc: str) -> None:
    """Adjacent windows overlap by exactly `overlap` characters before stripping."""
    n_chunks = len(chunk_text(sample_doc))
    for k in range(n_chunks - 1):
        a_start = k * DEFAULT_STRIDE
        b_start = (k + 1) * DEFAULT_STRIDE
        tail = sample_doc[a_start : a_start + DEFAULT_CHUNK_SIZE][-DEFAULT_OVERLAP:]
        head = sample_doc[b_start : b_start + DEFAULT_CHUNK_SIZE][:DEFAULT_OVERLAP]
        assert tail == head


def test_short_text_yields_one_chunk() -> None:
    text = "Python is a high-level language."
    assert chunk_text(text) == [{"text": text, "chunk_index": 0}]


def test_text_exactly_chunk_size_yields_one_chunk() -> None:
    text = "x" * 800
    chunks = chunk_text(text)
    assert len(chunks) == 1
    assert chunks[0]["text"] == text


def test_one_char_over_chunk_size_yields_a_151_char_tail() -> None:
    """A 1-character overflow produces a full-overlap second chunk, not a 1-char one.

    This is a real inefficiency in the current design — a document 801 characters long is
    stored as 800 + 151 characters, duplicating the overlap for almost no new content. It
    is pinned here so that a chunker which "fixes" it is recognised as a behavior change.
    """
    text = "x" * 801
    chunks = chunk_text(text)
    assert len(chunks) == 2
    assert len(chunks[0]["text"]) == 800
    assert len(chunks[1]["text"]) == 151


# --------------------------------------------------------------------------------------
# Stripping and index density
# --------------------------------------------------------------------------------------


def test_each_chunk_is_stripped() -> None:
    text = "   " + ("a" * 1000) + "   "
    for chunk in chunk_text(text):
        assert chunk["text"] == chunk["text"].strip()


def test_whitespace_only_windows_are_skipped_without_consuming_an_index() -> None:
    """Blank windows vanish, and `chunk_index` stays dense.

    `idx` only increments on append, so a skipped window does not leave a gap. This means
    `chunk_index` is a position in the *output list*, not a position in the document —
    which is precisely why chunk ids cannot be used as stable eval labels.
    """
    text = "ab" + (" " * 10) + "cd"  # len 14
    chunks = chunk_text(text, chunk_size=5, overlap=1)  # stride 4

    assert [c["text"] for c in chunks] == ["ab", "c", "cd"]
    assert [c["chunk_index"] for c in chunks] == [0, 1, 2]


def test_whitespace_only_text_yields_no_chunks() -> None:
    assert chunk_text("      \n\n\t  ") == []


def test_empty_text_yields_no_chunks() -> None:
    assert chunk_text("") == []


# --------------------------------------------------------------------------------------
# Contract
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(100, 100), (100, 150), (0, 0), (10, 10)],
)
def test_rejects_overlap_greater_or_equal_to_chunk_size(chunk_size: int, overlap: int) -> None:
    with pytest.raises(ValueError, match="chunk_size must be greater than overlap"):
        chunk_text("some text", chunk_size=chunk_size, overlap=overlap)


def test_empty_text_is_checked_before_invalid_parameters() -> None:
    """The empty-text guard runs first, so bad parameters go unreported on empty input."""
    assert chunk_text("", chunk_size=10, overlap=99) == []


def test_returns_exactly_text_and_chunk_index_keys(sample_doc: str) -> None:
    for chunk in chunk_text(sample_doc):
        assert set(chunk) == {"text", "chunk_index"}


def test_chunk_indices_are_contiguous_from_zero(sample_doc: str) -> None:
    chunks = chunk_text(sample_doc)
    assert [c["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_no_chunk_exceeds_chunk_size(sample_doc: str) -> None:
    for chunk in chunk_text(sample_doc):
        assert len(chunk["text"]) <= DEFAULT_CHUNK_SIZE
