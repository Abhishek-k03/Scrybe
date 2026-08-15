"""Compare result artifacts over the queries all of them could score.

    python evals/compare.py evals/results/*.json

A chunking change moves which gold spans fall inside a chunk, so two runs can disagree
about how many queries are answerable at all. Averaging each run over its own set and
putting the numbers side by side would compare different measurements. This restricts
every run to the intersection and says how many queries that left.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

HEADLINE = ("recall@5", "doc_recall@5", "ndcg@5", "mrr", "chars@5")


def scored_ids(artifact: dict[str, Any]) -> set[str]:
    return {row["id"] for row in artifact["per_query"] if row.get("scored")}


def mean_over(artifact: dict[str, Any], ids: set[str], metric: str) -> float | None:
    values = [
        row["metrics"][metric]
        for row in artifact["per_query"]
        if row["id"] in ids and "metrics" in row
    ]
    return sum(values) / len(values) if values else None


def name_of(path: Path, artifact: dict[str, Any]) -> str:
    config = artifact["config"]
    chunk = config["chunk"]
    size = chunk.get("chunk_size") or chunk.get("max_chars") or chunk.get("max_tokens")
    return f"{chunk['kind']}/{size} {config['retrieve']['kind']}+{config['rerank']['kind']}"


def report(paths: Sequence[Path]) -> str:
    artifacts = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]

    corpora = {artifact["corpus"]["sha256"] for _, artifact in artifacts}
    labels = {artifact["labels"]["sha256"] for _, artifact in artifacts}

    common = set.intersection(*(scored_ids(artifact) for _, artifact in artifacts))
    lines: list[str] = []

    if len(corpora) > 1:
        lines.append("!! these runs used different corpora; they are not comparable\n")
    if len(labels) > 1:
        lines.append("!! these runs used different labels; they are not comparable\n")

    widest = max(len(name_of(path, artifact)) for path, artifact in artifacts)
    header = f"{'config':<{widest}}  {'n':>3}  " + "  ".join(f"{m:>13}" for m in HEADLINE)
    lines.append(header)
    lines.append("-" * len(header))

    for path, artifact in artifacts:
        own = scored_ids(artifact)
        cells = []
        for metric in HEADLINE:
            value = mean_over(artifact, common, metric)
            if value is None:
                cells.append("          n/a")
            elif metric == "chars@5":
                cells.append(f"{round(value):>13,}")
            else:
                cells.append(f"{value:>13.4f}")
        marker = "" if own == common else f"  ({len(own)} scored on its own)"
        lines.append(
            f"{name_of(path, artifact):<{widest}}  {len(common):>3}  " + "  ".join(cells) + marker
        )

    lines.append("")
    lines.append(f"Averaged over the {len(common)} queries every run could score.")

    dropped = set.union(*(scored_ids(a) for _, a in artifacts)) - common
    if dropped:
        lines.append(
            f"Excluded because at least one run could not score them: {', '.join(sorted(dropped))}"
        )

    authors = {artifact["labels"]["author"] for _, artifact in artifacts}
    for author in sorted(authors):
        lines.append(f"Labels by: {author}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    paths = sorted(p for p in args.results if p.exists())
    if len(paths) < 2:
        print("need at least two result artifacts")
        return 2

    print(report(paths))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
