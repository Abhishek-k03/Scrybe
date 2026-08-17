"""Sweep the reranker's score floor and report what abstaining costs.

    python evals/abstention.py --config evals/configs/hybrid_rerank.json

Retrieval that returns nothing is the honest answer when the corpus cannot answer. Cosine
distance gave no defensible place to put that floor — its scale is a property of the
embedding space, not of the question. A cross-encoder score is query-conditioned, so a
threshold at least means something. Whether it means enough is what this measures.

Every threshold reads the same cached reranker scores, so the whole sweep after the first
threshold costs nothing (rule 3).
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
from app.rag.pipeline import build_pipeline  # noqa: E402
from evals import label_schema, metrics  # noqa: E402
from evals import run as harness  # noqa: E402
from evals.metrics import QueryOutcome  # noqa: E402

# Chosen to span the observed score range rather than fitted to the outcome; the grid is
# fixed here so a re-run cannot quietly move the points to flatter its own curve.
THRESHOLDS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35)


class NotReRankedError(ValueError):
    """Raised for a config whose rerank stage has no score to threshold."""


def with_threshold(config: PipelineConfig, threshold: float | None) -> PipelineConfig:
    if not hasattr(config.rerank, "score_threshold"):
        raise NotReRankedError(f"{config.rerank.kind} produces no score to threshold")
    return config.model_copy(
        update={"rerank": config.rerank.model_copy(update={"score_threshold": threshold})}
    )


def curve_row(threshold: float | None, outcomes: list[QueryOutcome], k: int) -> dict[str, Any]:
    """One point on the curve: what the floor caught, and what it cost to catch it."""
    summary = metrics.summarise(outcomes, [k])
    return {
        "score_threshold": threshold,
        "abstention_rate": summary["abstention_rate"],
        "false_abstention_rate": summary["false_abstention_rate"],
        f"recall@{k}": summary[f"recall@{k}"],
        f"ndcg@{k}": summary[f"ndcg@{k}"],
        "abstained_query_ids": sorted(o.query_id for o in outcomes if o.abstained),
    }


def format_curve(artifact: dict[str, Any]) -> str:
    k = artifact["run"]["k"]
    counts = artifact["run"]
    lines = [
        f"corpus     {counts['n_docs']} docs / {counts['n_chunks']} chunks",
        f"queries    {counts['n_answerable']} answerable, {counts['n_unanswerable']} unanswerable",
        f"reranker   {artifact['config']['rerank']['model']}",
        "",
        f"{'threshold':>10}  {'abstain(unans)':>15}  {'false abstain':>14}"
        f"  {'recall@' + str(k):>10}  {'ndcg@' + str(k):>9}",
        "-" * 68,
    ]
    def show(row: dict[str, Any], key: str) -> str:
        value = row[key]
        return "  n/a" if value is None else f"{value:.4f}"

    for row in artifact["curve"]:
        floor = "none" if row["score_threshold"] is None else f"{row['score_threshold']:.2f}"
        lines.append(
            f"{floor:>10}  {show(row, 'abstention_rate'):>15}"
            f"  {show(row, 'false_abstention_rate'):>14}"
            f"  {show(row, f'recall@{k}'):>10}  {show(row, f'ndcg@{k}'):>9}"
        )
    lines.append("")
    lines.append(
        f"  {counts['n_unanswerable']} unanswerable queries. Any floor picked to maximise the "
        "second column\n  against a sample that size is fitted to it, not validated on it."
    )
    return "\n".join(lines)


async def sweep(
    config: PipelineConfig,
    labels: label_schema.LabelSet,
    documents: list,
    api_key: str,
    thresholds: tuple[float, ...],
    k: int,
) -> dict[str, Any]:
    doc_ids = {doc.label: doc.doc_id for doc in documents}

    base = build_pipeline(with_threshold(config, None), api_key=api_key)
    report = await base.index_documents(documents)

    curve = [curve_row(None, await harness.evaluate(base, labels, doc_ids, k), k)]
    for threshold in thresholds:
        # Same index, same cached scores; only the floor moves.
        pipeline = build_pipeline(
            with_threshold(config, threshold), api_key=api_key, index=base.index
        )
        outcomes = await harness.evaluate(pipeline, labels, doc_ids, k)
        curve.append(curve_row(threshold, outcomes, k))

    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": harness.git_state(),
        "config": json.loads(with_threshold(config, None).to_json()),
        "run": {
            "k": k,
            "thresholds": list(thresholds),
            "n_docs": len(documents),
            "n_chunks": report.chunks_added,
            "n_answerable": sum(1 for q in labels.queries if q.answerable),
            "n_unanswerable": sum(1 for q in labels.queries if not q.answerable),
        },
        "labels": {
            "author": labels.author,
            "sha256": harness.file_sha256(harness.LABELS),
        },
        "curve": curve,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="PipelineConfig JSON")
    parser.add_argument("--labels", type=Path, default=harness.LABELS)
    parser.add_argument("--k", type=int, default=5, help="cut-off the curve is reported at")
    parser.add_argument("--dry-run", action="store_true", help="print without writing")
    parser.add_argument("--label", default="abstention", help="suffix for the artifact filename")
    args = parser.parse_args()

    labels_path = args.labels.resolve()
    if not labels_path.exists():
        print(f"no labels at {labels_path}")
        return 2

    config = harness.resolve_paths(PipelineConfig.from_file(args.config))
    try:
        with_threshold(config, None)
    except NotReRankedError as error:
        print(f"{error}; use a config with a cross-encoder rerank stage")
        return 2

    labels = label_schema.load(labels_path)
    documents = load_directory(harness.CORPUS_DIR)

    problems = label_schema.check(labels, documents, label_schema.corpus_hash(harness.MANIFEST))
    if problems:
        print("labels are inconsistent with the corpus; refusing to produce numbers:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    import os

    from dotenv import load_dotenv

    load_dotenv(REPO_ROOT / "backend" / ".env", override=True)
    api_key = os.environ.get("JINA_API_KEY", "")
    if not api_key:
        print("JINA_API_KEY is not set")
        return 2

    artifact = await sweep(config, labels, documents, api_key, THRESHOLDS, args.k)
    artifact["labels"]["sha256"] = harness.file_sha256(labels_path)
    artifact["labels"]["path"] = harness.repo_relative(labels_path)

    print(format_curve(artifact))

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
