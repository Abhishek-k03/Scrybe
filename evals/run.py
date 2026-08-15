"""Run one pipeline configuration against the labelled corpus and record the result.

    python evals/run.py                                  # the baseline config
    python evals/run.py --config evals/configs/hybrid.json
    python evals/run.py --offline                        # fake embedder, no network
    python evals/run.py --dry-run                        # print, write nothing

Every artifact under evals/results/ carries the full config, the git SHA, whether the tree
was dirty, the corpus and label hashes, and the sample sizes — so no number in it can be
read without knowing exactly what produced it.

Embeddings are cached on disk by content hash, so re-running a sweep costs nothing after
the first pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from app.rag.config import (  # noqa: E402
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    JinaEmbedConfig,
    MemoryIndexConfig,
    NoopRerankConfig,
    PipelineConfig,
)
from app.rag.ingest.local import load_directory  # noqa: E402
from app.rag.pipeline import Pipeline, build_pipeline  # noqa: E402
from evals import label_schema, metrics  # noqa: E402
from evals.label_schema import LabelSet  # noqa: E402
from evals.metrics import QueryOutcome  # noqa: E402

CORPUS_DIR = REPO_ROOT / "evals" / "corpus"
MANIFEST = REPO_ROOT / "evals" / "manifest.json"
LABELS = REPO_ROOT / "evals" / "labels" / "retrieval.json"
RESULTS_DIR = REPO_ROOT / "evals" / "results"
CACHE_DIR = REPO_ROOT / ".cache" / "embeddings"

K_VALUES = (1, 3, 5, 10)


def baseline_config() -> PipelineConfig:
    """What the app runs today, widened to the largest k the report needs."""
    return PipelineConfig(
        chunk=FixedCharChunkConfig(chunk_size=800, overlap=150),
        embed=JinaEmbedConfig(cache_dir=str(CACHE_DIR)),
        # Exact search: no ANN approximation, so a re-run cannot drift.
        index=MemoryIndexConfig(),
        retrieve=DenseRetrieveConfig(top_k=max(K_VALUES)),
        rerank=NoopRerankConfig(),
    )


def resolve_paths(config: PipelineConfig) -> PipelineConfig:
    """Make a relative cache_dir absolute against the repo, not the working directory."""
    if config.embed.kind != "jina" or config.embed.cache_dir is None:
        return config
    cache = Path(config.embed.cache_dir)
    if cache.is_absolute():
        return config
    return config.model_copy(
        update={"embed": config.embed.model_copy(update={"cache_dir": str(REPO_ROOT / cache)})}
    )


def git_state() -> dict[str, Any]:
    """The commit these numbers came from, and whether it fully describes the code."""

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
            ).stdout.strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    status = git("status", "--porcelain")
    return {
        "sha": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        # A dirty tree means the SHA does not reproduce this run.
        "dirty": bool(status),
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def evaluate(
    pipeline: Pipeline, labels: LabelSet, doc_ids: dict[str, str], top_k: int
) -> list[QueryOutcome]:
    indexed = pipeline.index.chunks()
    outcomes: list[QueryOutcome] = []

    for labelled in labels.queries:
        relevant = label_schema.relevant_chunks(indexed, labelled, doc_ids)
        result = await pipeline.retrieve(labelled.query, top_k=top_k)
        outcomes.append(
            QueryOutcome(
                query_id=labelled.id,
                answerable=labelled.answerable,
                retrieved_chunk_ids=tuple(hit.chunk.chunk_id for hit in result.hits),
                retrieved_doc_ids=tuple(hit.chunk.doc_id for hit in result.hits),
                relevant_chunk_ids=frozenset(chunk.chunk_id for chunk in relevant),
                relevant_doc_ids=frozenset(chunk.doc_id for chunk in relevant),
                retrieved_chunk_chars=tuple(len(hit.chunk.text) for hit in result.hits),
            )
        )
    return outcomes


def per_query_records(
    outcomes: Sequence[QueryOutcome], labels: LabelSet
) -> list[dict[str, Any]]:
    """One row per query, carrying its own scores.

    Every metric is stored per query, not just aggregated, so two runs can be compared over
    the subset of queries both were able to score — which is the only fair comparison when
    a chunking change moves which queries are reachable at all.
    """
    text_of = {labelled.id: labelled.query for labelled in labels.queries}
    rows: list[dict[str, Any]] = []

    for outcome in outcomes:
        row: dict[str, Any] = {
            "id": outcome.query_id,
            "query": text_of[outcome.query_id],
            "answerable": outcome.answerable,
            "scored": bool(outcome.answerable and outcome.relevant_chunk_ids),
            "abstained": outcome.abstained,
            "n_relevant": len(outcome.relevant_chunk_ids),
            "n_retrieved": len(outcome.retrieved_chunk_ids),
            "retrieved_chunk_chars": list(outcome.retrieved_chunk_chars),
            "retrieved": list(outcome.retrieved_chunk_ids),
            "relevant_found": [
                identifier
                for identifier in outcome.retrieved_chunk_ids
                if identifier in outcome.relevant_chunk_ids
            ],
        }
        if row["scored"]:
            scores: dict[str, float] = {
                "mrr": metrics.reciprocal_rank(
                    outcome.retrieved_chunk_ids, outcome.relevant_chunk_ids
                )
            }
            for k in K_VALUES:
                scores[f"recall@{k}"] = metrics.recall_at_k(
                    outcome.retrieved_chunk_ids, outcome.relevant_chunk_ids, k
                )
                scores[f"doc_recall@{k}"] = metrics.recall_at_k(
                    outcome.retrieved_doc_ids, outcome.relevant_doc_ids, k
                )
                scores[f"ndcg@{k}"] = metrics.ndcg_at_k(
                    outcome.retrieved_chunk_ids, outcome.relevant_chunk_ids, k
                )
                scores[f"hit@{k}"] = metrics.hit_at_k(
                    outcome.retrieved_chunk_ids, outcome.relevant_chunk_ids, k
                )
                scores[f"chars@{k}"] = float(outcome.chars_at_k(k))
            row["metrics"] = scores
        rows.append(row)
    return rows


def build_artifact(
    config: PipelineConfig,
    labels: LabelSet,
    outcomes: Sequence[QueryOutcome],
    n_docs: int,
    n_chunks: int,
    top_k: int,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_state(),
        "config": json.loads(config.to_json()),
        "run": {
            "retrieved_top_k": top_k,
            "k_values": list(K_VALUES),
            "index_kind": config.index.kind,
            "embed_model": getattr(config.embed, "model", config.embed.kind),
        },
        "corpus": {
            "sha256": label_schema.corpus_hash(MANIFEST),
            "n_docs": n_docs,
            "n_chunks": n_chunks,
        },
        "labels": {
            "path": LABELS.relative_to(REPO_ROOT).as_posix(),
            "sha256": file_sha256(LABELS),
            "author": labels.author,
            "caveats": label_schema.warnings(labels),
        },
        "metrics": metrics.summarise(outcomes, K_VALUES),
        "per_query": per_query_records(outcomes, labels),
    }


def format_summary(artifact: dict[str, Any]) -> str:
    summary: dict[str, Any] = artifact["metrics"]
    lines = [
        f"corpus     {artifact['corpus']['n_docs']} docs / "
        f"{artifact['corpus']['n_chunks']} chunks",
        f"queries    {summary['n_answerable']} answerable, "
        f"{summary['n_unanswerable']} unanswerable",
        f"config     {artifact['run']['embed_model']} + {artifact['config']['retrieve']['kind']}"
        f" + {artifact['config']['rerank']['kind']}, top_k={artifact['run']['retrieved_top_k']}",
        "",
    ]
    if summary.get("n_unreachable"):
        lines.append(
            f"  !! {summary['n_unreachable']} answerable quer(y/ies) have no matching chunk at "
            f"this chunk size and are excluded: {', '.join(summary['unreachable_query_ids'])}"
        )
        lines.append(
            f"     scores below are over {summary['n_scored']} queries, not "
            f"{summary['n_answerable']} — do not compare them to a run with a different count"
        )
        lines.append("")

    for key, value in summary.items():
        if key.startswith("n_") or key == "unreachable_query_ids":
            continue
        shown = "n/a" if value is None else f"{value:.4f}"
        lines.append(f"  {key:<24} {shown}")
    return "\n".join(lines)


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="PipelineConfig JSON")
    parser.add_argument(
        "--offline", action="store_true", help="use the fake embedder; no network, no key"
    )
    parser.add_argument("--dry-run", action="store_true", help="print without writing")
    parser.add_argument("--label", default="", help="suffix for the artifact filename")
    args = parser.parse_args()

    if not LABELS.exists():
        print(f"no labels at {LABELS}")
        return 2

    config = PipelineConfig.from_file(args.config) if args.config else baseline_config()
    if args.offline:
        config = config.model_copy(update={"embed": FakeEmbedConfig(dimensions=512)})
    config = resolve_paths(config)

    labels = label_schema.load(LABELS)
    documents = load_directory(CORPUS_DIR)

    problems = label_schema.check(labels, documents, label_schema.corpus_hash(MANIFEST))
    if problems:
        print("labels are inconsistent with the corpus; refusing to produce numbers:")
        for problem in problems:
            print(f"  {problem}")
        return 1

    api_key = ""
    if config.embed.kind == "jina":
        import os

        from dotenv import load_dotenv

        load_dotenv(REPO_ROOT / "backend" / ".env", override=True)
        api_key = os.environ.get("JINA_API_KEY", "")
        if not api_key:
            print("JINA_API_KEY is not set; use --offline for a run without it")
            return 2

    pipeline = build_pipeline(config, api_key=api_key)
    report = await pipeline.index_documents(documents)
    doc_ids = {doc.label: doc.doc_id for doc in documents}

    top_k = config.retrieve.top_k
    outcomes = await evaluate(pipeline, labels, doc_ids, top_k)
    artifact = build_artifact(
        config, labels, outcomes, len(documents), report.chunks_added, top_k
    )

    print(format_summary(artifact))
    for caveat in artifact["labels"]["caveats"]:
        print(f"\n  CAVEAT  {caveat}")

    if args.dry_run:
        print("\ndry run — nothing written")
        return 0

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = f"-{args.label}" if args.label else ""
    destination = RESULTS_DIR / f"{stamp}{suffix}.json"
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {destination.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    import asyncio

    raise SystemExit(asyncio.run(main()))
