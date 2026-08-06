"""Shared test fixtures.

The overriding concern here is isolation. `backend/chroma_db/` is the live demo index and
the only copy of it — it is gitignored, so an accidental write is not recoverable with
`git checkout`. Two autouse fixtures enforce that no test can reach it:

* `_isolated_chroma` (per test) points `settings.CHROMA_PATH` at a throwaway directory and
  resets the module-level singletons in `app.services.store` on both sides of the test.
* `_live_index_untouched` (per session) hashes the real index before and after the whole
  run and fails loudly if a single byte moved.

The second exists because the first is easy to defeat by accident: `store._get_collection()`
caches its client in a module global, so any module that grabbed a handle before the
monkeypatch would keep writing to wherever it was first pointed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
LIVE_CHROMA_DIR = BACKEND_DIR / "chroma_db"


def _fingerprint_dir(root: Path) -> dict[str, str]:
    """Map every file under `root` to a hash of its contents."""
    if not root.exists():
        return {}
    out: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            out[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


@pytest.fixture(scope="session", autouse=True)
def _live_index_untouched() -> None:
    """Fail the run if the live ChromaDB index changed while tests executed."""
    before = _fingerprint_dir(LIVE_CHROMA_DIR)
    yield
    after = _fingerprint_dir(LIVE_CHROMA_DIR)

    if before == after:
        return

    changed = sorted(
        set(before) ^ set(after) | {k for k in set(before) & set(after) if before[k] != after[k]}
    )
    pytest.fail(
        "The live ChromaDB index at backend/chroma_db/ was modified by the test suite.\n"
        "It is gitignored, so this is not recoverable via git.\n"
        f"Changed files: {changed}"
    )


@pytest.fixture(autouse=True)
def _isolated_chroma(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point ChromaDB at a per-test temp directory and clear cached singletons."""
    from app.core.config import settings
    from app.services import store

    chroma_dir = tmp_path / "chroma"
    monkeypatch.setattr(settings, "CHROMA_PATH", str(chroma_dir), raising=False)

    # Reset on entry as well as exit: a previous test may have populated these.
    monkeypatch.setattr(store, "_client", None, raising=False)
    monkeypatch.setattr(store, "_collection", None, raising=False)

    yield chroma_dir

    store._client = None
    store._collection = None


@pytest.fixture(autouse=True)
def _no_accidental_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Blank the API keys so an unmocked embed/LLM call raises instead of billing.

    `embed_texts` and `generate_answer` both check for a key and raise RuntimeError when
    it is missing, so this turns a silent paid call into an immediate, obvious failure.
    Tests that genuinely need a key should be marked `@pytest.mark.live`.
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "JINA_API_KEY", "", raising=False)
    monkeypatch.setattr(settings, "GROQ_API_KEY", "", raising=False)
