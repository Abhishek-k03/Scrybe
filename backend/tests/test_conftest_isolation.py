"""Tests for the safety fixtures themselves — an unverified guard gets trusted anyway."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from tests.conftest import LIVE_CHROMA_DIR, stored_chunks_fingerprint


def test_chroma_path_is_redirected_away_from_the_live_index() -> None:
    from app.core.config import settings

    configured = Path(settings.CHROMA_PATH).resolve()
    assert configured != LIVE_CHROMA_DIR.resolve()


def test_pipeline_singletons_start_unset() -> None:
    from app.services import pipeline

    assert pipeline._index is None
    assert pipeline._pipeline is None


def test_api_keys_are_blank_so_unmocked_calls_fail_loudly() -> None:
    from app.core.config import settings

    assert settings.JINA_API_KEY == ""
    assert settings.GROQ_API_KEY == ""


def test_writing_through_the_store_lands_in_the_temp_dir(_isolated_chroma: Path) -> None:
    from app.rag.types import Chunk
    from app.services.store import add_chunks, count_total_chunks

    chunk = Chunk(
        doc_id="test-source",
        doc_label="fixture",
        source_type="test",
        chunk_index=0,
        text="hello",
        start_char=0,
        end_char=5,
    )
    written = add_chunks([chunk], [[0.1, 0.2, 0.3]])

    assert written == 1
    assert count_total_chunks() == 1
    assert _isolated_chroma.exists(), "ChromaDB did not write to the temp directory"


# --------------------------------------------------------------------------------------
# The fingerprint guard
# --------------------------------------------------------------------------------------


def _make_index(path: Path, rows: list[tuple[str, str, str]]) -> None:
    """Build a minimal Chroma-shaped sqlite file."""
    path.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path / "chroma.sqlite3")
    con.execute("CREATE TABLE embeddings (id INTEGER PRIMARY KEY, embedding_id TEXT)")
    con.execute(
        "CREATE TABLE embedding_metadata (id INTEGER, key TEXT, string_value TEXT, "
        "int_value INTEGER, float_value REAL)"
    )
    for i, (embedding_id, key, value) in enumerate(rows):
        con.execute("INSERT INTO embeddings VALUES (?, ?)", (i, embedding_id))
        con.execute(
            "INSERT INTO embedding_metadata VALUES (?, ?, ?, NULL, NULL)", (i, key, value)
        )
    con.commit()
    con.close()


def test_fingerprint_detects_changed_chunk_content(tmp_path: Path) -> None:
    _make_index(tmp_path, [("chunk-0", "chroma:document", "hello")])
    before = stored_chunks_fingerprint(tmp_path)

    (tmp_path / "chroma.sqlite3").unlink()
    _make_index(tmp_path, [("chunk-0", "chroma:document", "goodbye")])
    assert stored_chunks_fingerprint(tmp_path) != before


def test_fingerprint_detects_added_chunks(tmp_path: Path) -> None:
    _make_index(tmp_path, [("chunk-0", "chroma:document", "hello")])
    before = stored_chunks_fingerprint(tmp_path)

    (tmp_path / "chroma.sqlite3").unlink()
    _make_index(
        tmp_path,
        [("chunk-0", "chroma:document", "hello"), ("chunk-1", "chroma:document", "world")],
    )
    assert stored_chunks_fingerprint(tmp_path) != before


def test_fingerprint_is_stable_across_repeated_calls(tmp_path: Path) -> None:
    _make_index(tmp_path, [("chunk-0", "chroma:document", "hello")])
    assert stored_chunks_fingerprint(tmp_path) == stored_chunks_fingerprint(tmp_path)


def test_fingerprint_of_missing_index_is_none(tmp_path: Path) -> None:
    assert stored_chunks_fingerprint(tmp_path / "nope") is None


def test_live_index_is_actually_being_watched() -> None:
    """Guards against the guard silently watching an empty or wrong path."""
    assert LIVE_CHROMA_DIR.exists(), "live index missing — the session guard would be a no-op"
    assert stored_chunks_fingerprint(LIVE_CHROMA_DIR) is not None
