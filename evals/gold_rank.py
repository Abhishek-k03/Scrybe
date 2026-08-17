"""Where do the gold chunks actually rank, and what could a perfect reranker recover?

    python evals/gold_rank.py --config evals/configs/dense_baseline.json

`recall@5` says a query failed. It does not say whether the gold chunk was at rank 6 or
rank 400, and those call for opposite fixes: the first is a ranking problem a reranker can
solve, the second is an embedding problem it cannot. This ranks every gold chunk against
the whole corpus and reports the ceiling a perfect reranker would hit at each pool width.

The ceiling is an upper bound, not a prediction. It assumes a reranker that always puts the
gold chunks first — no real one does. It is useful for deciding whether reranking is worth
building at all, and for reading the measured result against what was available.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from app.rag.config import PipelineConfig  # noqa: E402
from app.rag.ingest.local import load_directory  # noqa: E402
from app.rag.pipeline import Pipeline, build_pipeline  # noqa: E402
from evals import label_schema  # noqa: E402
from evals import run as harness  # noqa: E402
from evals.label_schema import LabelSet  # noqa: E402

POOL_WIDTHS = (5, 10, 20, 50, 100)


def ranks_of(retrieved: list[str], relevant: frozenset[str]) -> list[int]:
    """1-based positions of the relevant chunks in the full ordering, ascending."""
    return sorted(i + 1 for i, chunk_id in enumerate(retrieved) if chunk_id in relevant)


def ceiling_at(ranks: list[int], n_relevant: int, pool: int, k: int) -> float:
    """Recall@k a reranker would reach over a pool of `pool` candidates, if it were perfect.

    A perfect reranker promotes every gold chunk inside the pool to the front, so at most
    `k` of them can land in the top k. Gold chunks outside the pool are unreachable: the
    reranker never sees them.
    """
    if not n_relevant:
        return 0.0
    reachable = sum(1 for rank in ranks if rank <= pool)
    return min(reachable, k) / n_relevant


async def measure(
    pipeline: Pipeline, labels: LabelSet, doc_ids: dict[str, str], k: int
) -> list[dict[str, Any]]:
    indexed = pipeline.index.chunks()
    # Rank against the whole corpus: a gold chunk at position 352 has to be visible as 352,
    # not as "absent", or the diagnostic cannot tell a ranking failure from a retrieval one.
    corpus_size = len(indexed)
    rows: list[dict[str, Any]] = []

    for labelled in labels.queries:
        if not labelled.answerable:
            continue
        relevant = label_schema.relevant_chunks(indexed, labelled, doc_ids)
        relevant_ids = frozenset(chunk.chunk_id for chunk in relevant)
        if not relevant_ids:
            continue

        result = await pipeline.retrieve(labelled.query, top_k=corpus_size)
        retrieved = [hit.chunk.chunk_id for hit in result.hits]
        ranks = ranks_of(retrieved, relevant_ids)

        rows.append(
            {
                "id": labelled.id,
                "query": labelled.query,
                "n_relevant": len(relevant_ids),
                "gold_ranks": ranks,
                "n_unranked": len(relevant_ids) - len(ranks),
                f"recall@{k}": ceiling_at(ranks, len(relevant_ids), k, k),
                "ceiling": {
                    str(pool): ceiling_at(ranks, len(relevant_ids), pool, k)
                    for pool in POOL_WIDTHS
                },
            }
        )
    return rows


def summarise(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    if not rows:
        return {"n_queries": 0}
    return {
        "n_queries": len(rows),
        "corpus_rank_ceiling": {
            pool: sum(row["ceiling"][pool] for row in rows) / len(rows)
            for pool in (str(p) for p in POOL_WIDTHS)
        },
        "worst_gold_rank": max(max(row["gold_ranks"], default=0) for row in rows),
    }


def format_report(artifact: dict[str, Any], k: int) -> str:
    rows = artifact["per_query"]
    summary = artifact["summary"]
    lines = [
        f"corpus     {artifact['run']['n_chunks']} chunks",
        f"queries    {summary['n_queries']} answerable and reachable",
        "",
        "Queries whose gold chunks are not all in the top 5 — the ones a reranker could fix:",
        "",
        f"  {'query':<12}  {'gold ranks (of ' + str(artifact['run']['n_chunks']) + ')':<28}  n_gold",
    ]
    for row in rows:
        if row[f"recall@{k}"] >= 1.0:
            continue
        ranks = ", ".join(str(r) for r in row["gold_ranks"]) or "not retrieved"
        lines.append(f"  {row['id']:<12}  {ranks:<28}  {row['n_relevant']}")

    lines += ["", f"Ceiling on recall@{k} with a perfect reranker over a pool of:", ""]
    for pool, value in summary["corpus_rank_ceiling"].items():
        lines.append(f"  fetch_k={pool:<5} {value:.4f}")
    lines += [
        "",
        "  An upper bound assuming a reranker that never makes a mistake. Read the measured",
        "  numbers in evals/results/ against it, not as if it were a target.",
    ]
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="PipelineConfig JSON")
    parser.add_argument("--labels", type=Path, default=harness.LABELS)
    parser.add_argument("--k", type=int, default=5, help="cut-off the ceiling is reported at")
    parser.add_argument("--dry-run", action="store_true", help="print without writing")
    parser.add_argument("--label", default="gold-rank", help="suffix for the artifact filename")
    args = parser.parse_args()

    labels_path = args.labels.resolve()
    if not labels_path.exists():
        print(f"no labels at {labels_path}")
        return 2

    config = PipelineConfig.from_file(args.config) if args.config else harness.baseline_config()
    config = harness.resolve_paths(config)

    labels = label_schema.load(labels_path)
    documents = load_directory(harness.CORPUS_DIR)

    problems = label_schema.check(labels, documents, label_schema.corpus_hash(harness.MANIFEST))
    if problems:
        print("labels are inconsistent with the corpus; refusing to produce numbers:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    api_key = ""
    if harness.needs_api_key(config):
        import os

        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / "backend" / ".env", override=True)
        api_key = os.environ.get("JINA_API_KEY", "")
        if not api_key:
            print("JINA_API_KEY is not set")
            return 2

    pipeline = build_pipeline(config, api_key=api_key)
    report = await pipeline.index_documents(documents)
    doc_ids = {doc.label: doc.doc_id for doc in documents}

    rows = await measure(pipeline, labels, doc_ids, args.k)
    artifact = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": harness.git_state(),
        "config": json.loads(config.to_json()),
        "run": {
            "k": args.k,
            "pool_widths": list(POOL_WIDTHS),
            "n_docs": len(documents),
            "n_chunks": report.chunks_added,
        },
        "labels": {
            "author": labels.author,
            "sha256": harness.file_sha256(labels_path),
            "path": harness.repo_relative(labels_path),
        },
        "summary": summarise(rows, args.k),
        "per_query": rows,
    }

    print(format_report(artifact, args.k))

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    harness.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    destination = harness.RESULTS_DIR / f"{stamp}-{args.label}.json"
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {destination.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
