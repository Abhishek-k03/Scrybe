"""Tests for the safety fixtures themselves.

An unverified guard is worse than no guard, because it is trusted. These confirm that the
autouse fixtures in conftest.py actually redirect ChromaDB away from the live index, and
that the session-level fingerprint check can in fact detect a modification.
"""

from __future__ import annotations

from pathlib import Path

from tests.conftest import LIVE_CHROMA_DIR, _fingerprint_dir


def test_chroma_path_is_redirected_away_from_the_live_index() -> None:
    from app.core.config import settings

    configured = Path(settings.CHROMA_PATH).resolve()
    assert configured != LIVE_CHROMA_DIR.resolve()
    assert "chroma_db" not in configured.parts


def test_store_singletons_start_unset() -> None:
    """If these leaked between tests, a cached client could outlive its monkeypatch."""
    from app.services import store

    assert store._client is None
    assert store._collection is None


def test_api_keys_are_blank_so_unmocked_calls_fail_loudly() -> None:
    from app.core.config import settings

    assert settings.JINA_API_KEY == ""
    assert settings.GROQ_API_KEY == ""


def test_writing_through_the_store_lands_in_the_temp_dir(_isolated_chroma: Path) -> None:
    """A real ChromaDB write must materialise under tmp_path, not in backend/chroma_db/."""
    from app.services.store import add_chunks, count_total_chunks

    written = add_chunks(
        source_id="test-source",
        source_label="fixture",
        source_type="test",
        chunks=[{"text": "hello", "chunk_index": 0}],
        embeddings=[[0.1, 0.2, 0.3]],
    )

    assert written == 1
    assert count_total_chunks() == 1
    assert _isolated_chroma.exists(), "ChromaDB did not write to the temp directory"


# --------------------------------------------------------------------------------------
# The fingerprint guard
# --------------------------------------------------------------------------------------


def test_fingerprint_detects_a_single_changed_byte(tmp_path: Path) -> None:
    target = tmp_path / "index.bin"
    target.write_bytes(b"aaaa")
    before = _fingerprint_dir(tmp_path)

    target.write_bytes(b"aaab")
    assert _fingerprint_dir(tmp_path) != before


def test_fingerprint_detects_added_and_removed_files(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"a")
    before = _fingerprint_dir(tmp_path)

    (tmp_path / "b.bin").write_bytes(b"b")
    after_add = _fingerprint_dir(tmp_path)
    assert after_add != before

    (tmp_path / "b.bin").unlink()
    assert _fingerprint_dir(tmp_path) == before


def test_fingerprint_of_missing_directory_is_empty(tmp_path: Path) -> None:
    assert _fingerprint_dir(tmp_path / "does-not-exist") == {}


def test_fingerprint_is_stable_across_repeated_calls(tmp_path: Path) -> None:
    (tmp_path / "a.bin").write_bytes(b"contents")
    assert _fingerprint_dir(tmp_path) == _fingerprint_dir(tmp_path)


def test_live_index_is_actually_being_watched() -> None:
    """Guards against the guard silently watching an empty/wrong path."""
    assert LIVE_CHROMA_DIR.name == "chroma_db"
    assert LIVE_CHROMA_DIR.exists(), "live index missing — the session guard would be a no-op"
    assert len(_fingerprint_dir(LIVE_CHROMA_DIR)) > 0
