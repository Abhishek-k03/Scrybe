"""Record the current top-k retrieval results against the live index.

Run:  python evals/record_golden.py

Writes backend/tests/golden/retrieval_top5.json, which the test suite replays to prove a
refactor did not change what retrieval returns. Costs one Jina query embedding per query
and only reads from ChromaDB; the index is fingerprinted before and after to prove it.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
BACKEND = REPO_ROOT / "backend"
OUT_PATH = BACKEND / "tests" / "golden" / "retrieval_top5.json"

# settings reads env_file=".env" and CHROMA_PATH="./chroma_db", both relative to the
# working directory, so both are resolved explicitly here rather than trusting the CWD.
load_dotenv(BACKEND / ".env")
sys.path.insert(0, str(BACKEND))

from app.core.config import settings  # noqa: E402

settings.CHROMA_PATH = str(BACKEND / "chroma_db")

from app.services.retriever import retrieve  # noqa: E402
from app.services.store import count_total_chunks  # noqa: E402

TOP_K = 5

QUERIES = [
    "When was Counter-Strike 2 released?",
    "What game engine does Counter-Strike 2 run on?",
    "How do the volumetric smoke grenades work?",
    "What happened to CS:GO when Counter-Strike 2 launched?",
    "What is sub-tick update architecture?",
    "Which maps were rebuilt or overhauled?",
    "How was the game received by reviewers?",
    "What weapon skins carried over from the previous game?",
    "What is the atmospheric scattering model?",
    "How does low light dehazing differ from daytime dehazing?",
    "How is the illumination map estimated?",
    "What is a transmission map?",
    "Which datasets were used for evaluation?",
    "What quantitative metrics were reported?",
    "What limitations does the method have?",
]


def fingerprint(chroma_dir: Path) -> str:
    """Hash the stored chunks: ids, documents and metadata.

    Compares logical content rather than file bytes — Chroma rewrites its SQLite pages and
    HNSW segment files whenever the index is opened, so the bytes change on a pure read.
    """
    db = chroma_dir / "chroma.sqlite3"
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT e.embedding_id, m.key, "
            "COALESCE(m.string_value, CAST(m.int_value AS TEXT), "
            "CAST(m.float_value AS TEXT), '') "
            "FROM embeddings e JOIN embedding_metadata m ON m.id = e.id "
            "ORDER BY e.embedding_id, m.key"
        ).fetchall()
    finally:
        con.close()
    payload = "\n".join("\t".join(str(v) for v in row) for row in rows)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


async def main() -> int:
    if not settings.JINA_API_KEY:
        print("JINA_API_KEY not set — cannot embed queries.", file=sys.stderr)
        return 1

    chroma_dir = Path(settings.CHROMA_PATH)
    before = fingerprint(chroma_dir)

    total_chunks = count_total_chunks()
    print(f"index: {total_chunks} chunks at {chroma_dir}")

    records = []
    for query in QUERIES:
        hits = await retrieve(query, top_k=TOP_K)
        records.append(
            {
                "query": query,
                "hits": [
                    {
                        "source_id": h["source_id"],
                        "source_label": h["source_label"],
                        "chunk_index": h["chunk_index"],
                        "distance": h["distance"],
                        "text_sha256": hashlib.sha256(h["text"].encode("utf-8")).hexdigest(),
                    }
                    for h in hits
                ],
            }
        )
        top = hits[0] if hits else None
        label = top["source_label"][:34] if top else "-"
        dist = f"{top['distance']:.4f}" if top else "-"
        print(f"  {len(hits)} hits  d={dist}  {label:36s} {query[:44]}")

    after = fingerprint(chroma_dir)
    if before != after:
        print("ABORT: the stored chunks changed during recording.", file=sys.stderr)
        return 1

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps(
            {
                "recorded_at": datetime.now(UTC).isoformat(),
                "embedding_model": "jina-embeddings-v3",
                "query_task": "retrieval.query",
                "collection": settings.COLLECTION_NAME,
                "index_chunk_count": total_chunks,
                "top_k": TOP_K,
                "queries": records,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {OUT_PATH.relative_to(REPO_ROOT)} — index verified unchanged")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
