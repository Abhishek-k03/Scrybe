"""Shared test fixtures.

`backend/chroma_db/` is gitignored and is the only copy of the live index, so a stray write
is unrecoverable. `_isolated_chroma` repoints ChromaDB at tmp_path per test;
`_live_index_untouched` hashes the stored chunks around the session as a backstop, since
`store` caches its client in a module global that can outlive a monkeypatch.

Tests marked `live` opt out of the redirection and run against the real index and real keys.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
LIVE_CHROMA_DIR = BACKEND_DIR / "chroma_db"


def stored_chunks_fingerprint(chroma_dir: Path) -> str | None:
    """Hash the stored chunk ids and metadata, or None if there is no index.

    Hashes logical content rather than file bytes: Chroma rewrites its SQLite pages and HNSW
    segment files whenever the index is opened, so the bytes change even on a pure read.
    """
    db = chroma_dir / "chroma.sqlite3"
    if not db.exists():
        return None
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT e.embedding_id, m.key, "
            "COALESCE(m.string_value, CAST(m.int_value AS TEXT), CAST(m.float_value AS TEXT), '') "
            "FROM embeddings e JOIN embedding_metadata m ON m.id = e.id "
            "ORDER BY e.embedding_id, m.key"
        ).fetchall()
    finally:
        con.close()
    payload = "\n".join("\t".join(str(v) for v in row) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@pytest.fixture(scope="session", autouse=True)
def _live_index_untouched() -> None:
    """Fail the run if the chunks stored in the live index changed."""
    before = stored_chunks_fingerprint(LIVE_CHROMA_DIR)
    yield
    after = stored_chunks_fingerprint(LIVE_CHROMA_DIR)
    if before != after:
        pytest.fail(
            "The live ChromaDB index at backend/chroma_db/ was modified by the test suite. "
            "It is gitignored, so this is not recoverable via git."
        )


@pytest.fixture(autouse=True)
def _isolated_chroma(request: pytest.FixtureRequest, tmp_path: Path, monkeypatch) -> Path | None:
    """Point ChromaDB at a per-test temp directory and clear cached singletons."""
    if request.node.get_closest_marker("live"):
        yield None
        return

    from app.core.config import settings
    from app.services import store

    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(settings, "CHROMA_PATH", str(chroma_dir), raising=False)
    monkeypatch.setattr(store, "_client", None, raising=False)
    monkeypatch.setattr(store, "_collection", None, raising=False)

    yield chroma_dir

    store._client = None
    store._collection = None


@pytest.fixture(autouse=True)
def _no_accidental_network(request: pytest.FixtureRequest, monkeypatch) -> None:
    """Blank the API keys so an unmocked embed/LLM call raises instead of billing."""
    if request.node.get_closest_marker("live"):
        return

    from app.core.config import settings

    monkeypatch.setattr(settings, "JINA_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "", raising=False)
