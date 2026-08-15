"""Replay recorded queries against the live index and check retrieval is unchanged.

Deselected by default; run with `pytest -m live`. Needs JINA_API_KEY and the real
backend/chroma_db/. Regenerate with `python evals/record_golden.py`.

Rankings are compared exactly, distances only within a tolerance: Chroma is bit-stable for
a fixed query vector, but embeddings returned for the same text have been observed to drift
by ~4e-4 between sessions, which moves distances without reordering hits.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

GOLDEN_PATH = Path(__file__).parent / "golden" / "retrieval_top5.json"
DISTANCE_TOLERANCE = 1e-3

pytestmark = pytest.mark.live


@pytest.fixture(scope="module")
def golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip("no golden recorded — run `python evals/record_golden.py`")
    return json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def live_settings() -> None:
    """Point at the real index and real key, overriding the isolation fixtures."""
    from dotenv import load_dotenv

    backend = Path(__file__).resolve().parents[1]
    load_dotenv(backend / ".env", override=True)

    import os

    from app.core.config import settings

    settings.CHROMA_PATH = str(backend / "chroma_db")
    settings.JINA_API_KEY = os.environ.get("JINA_API_KEY", "")
    if not settings.JINA_API_KEY:
        pytest.skip("JINA_API_KEY not set")


_replay_cache: dict[tuple[str, int], list[dict]] = {}


async def _replay(query: str, top_k: int) -> list[dict]:
    """Retrieve once per (query, k) so the whole module costs one embedding per query."""
    key = (query, top_k)
    if key not in _replay_cache:
        from app.services import pipeline
        from app.services.retriever import retrieve

        pipeline.reset()
        _replay_cache[key] = await retrieve(query, top_k=top_k)
    return _replay_cache[key]


async def test_index_still_has_the_recorded_chunk_count(golden: dict, live_settings: None) -> None:
    from app.services import pipeline
    from app.services.store import count_total_chunks

    pipeline.reset()
    assert count_total_chunks() == golden["index_chunk_count"]


async def test_recorded_queries_return_the_same_ranking(golden: dict, live_settings: None) -> None:
    """The ordered list of chunks per query must be identical."""
    top_k = golden["top_k"]
    drifted: list[str] = []

    for record in golden["queries"]:
        hits = await _replay(record["query"], top_k)
        actual = [(h["source_id"], h["chunk_index"]) for h in hits]
        expected = [(h["source_id"], h["chunk_index"]) for h in record["hits"]]
        if actual != expected:
            drifted.append(f"{record['query']!r}\n    expected {expected}\n    actual   {actual}")

    assert not drifted, "retrieval ranking changed:\n  " + "\n  ".join(drifted)


async def test_recorded_queries_return_the_same_chunk_text(
    golden: dict, live_settings: None
) -> None:
    top_k = golden["top_k"]
    for record in golden["queries"]:
        hits = await _replay(record["query"], top_k)
        for hit, expected in zip(hits, record["hits"], strict=True):
            actual = hashlib.sha256(hit["text"].encode("utf-8")).hexdigest()
            assert actual == expected["text_sha256"], f"chunk text changed for {record['query']!r}"


async def test_distances_are_within_tolerance(golden: dict, live_settings: None) -> None:
    top_k = golden["top_k"]
    for record in golden["queries"]:
        hits = await _replay(record["query"], top_k)
        for hit, expected in zip(hits, record["hits"], strict=True):
            delta = abs(hit["distance"] - expected["distance"])
            assert delta < DISTANCE_TOLERANCE, (
                f"{record['query']!r}: distance moved {delta:.6f} "
                f"({expected['distance']:.6f} -> {hit['distance']:.6f})"
            )
