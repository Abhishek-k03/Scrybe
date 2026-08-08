"""Integrity checks for the evaluation corpus.

Corpus drift invalidates every metric measured against it silently — the numbers still
compute, they just stop being comparable. These fail if the files diverge from manifest.json.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALS_DIR = REPO_ROOT / "evals"
MANIFEST_PATH = EVALS_DIR / "manifest.json"

# Below this, recall@k says more about the corpus size than about the retriever.
MIN_DOCUMENTS = 20


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST_PATH.exists():
        pytest.skip("corpus not fetched — run `python evals/fetch_corpus.py`")
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_every_document_file_exists(manifest: dict) -> None:
    for doc in manifest["documents"]:
        assert (EVALS_DIR / doc["file"]).exists(), f"missing corpus file: {doc['file']}"


def test_every_document_matches_its_recorded_hash(manifest: dict) -> None:
    """Detects an edited or re-fetched article that was not re-manifested."""
    drifted = []
    for doc in manifest["documents"]:
        text = (EVALS_DIR / doc["file"]).read_text(encoding="utf-8")
        actual = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if actual != doc["sha256"]:
            drifted.append(doc["slug"])
    assert not drifted, f"corpus files changed without a manifest update: {drifted}"


def test_corpus_hash_is_reproducible_from_the_documents(manifest: dict) -> None:
    """The aggregate hash recorded in eval artifacts must derive from the parts."""
    expected = hashlib.sha256(
        "".join(d["sha256"] for d in manifest["documents"]).encode("utf-8")
    ).hexdigest()
    assert manifest["corpus_sha256"] == expected


def test_recorded_char_counts_match_the_files(manifest: dict) -> None:
    for doc in manifest["documents"]:
        text = (EVALS_DIR / doc["file"]).read_text(encoding="utf-8")
        assert len(text) == doc["chars"], f"{doc['slug']}: char count drifted"


def test_document_count_matches_manifest(manifest: dict) -> None:
    assert manifest["document_count"] == len(manifest["documents"])


def test_corpus_is_large_enough_to_evaluate(manifest: dict) -> None:
    assert manifest["document_count"] >= MIN_DOCUMENTS, (
        f"{manifest['document_count']} documents is too few for meaningful retrieval metrics"
    )


def test_slugs_are_unique(manifest: dict) -> None:
    slugs = [d["slug"] for d in manifest["documents"]]
    assert len(slugs) == len(set(slugs))


def test_every_document_pins_a_revision_id(manifest: dict) -> None:
    """Without a revision id the source is not re-fetchable, so the corpus is not reproducible."""
    for doc in manifest["documents"]:
        assert doc["revision_id"] > 0, f"{doc['slug']} has no pinned revision"


def test_no_document_is_trivially_short(manifest: dict) -> None:
    """A stub article contributes noise and no answerable questions."""
    stubs = [d["slug"] for d in manifest["documents"] if d["chars"] < 500]
    assert not stubs, f"stub articles should be dropped from the corpus: {stubs}"
