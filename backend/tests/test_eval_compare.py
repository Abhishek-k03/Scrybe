"""The side-by-side comparison table.

Its whole job is to stop two runs being compared over different query sets, so the tests
are mostly about what it refuses to average and what it says out loud when it drops one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from evals import compare


def artifact(
    *,
    scored: dict[str, dict[str, float]],
    unreachable: tuple[str, ...] = (),
    chunk: dict[str, Any] | None = None,
    retrieve: dict[str, Any] | None = None,
    rerank: dict[str, Any] | None = None,
    corpus_sha: str = "corpus-a",
    labels_sha: str = "labels-a",
    author: str = "abhi",
) -> dict[str, Any]:
    rows = [{"id": qid, "scored": True, "metrics": m} for qid, m in scored.items()]
    rows += [{"id": qid, "scored": False} for qid in unreachable]
    return {
        "config": {
            "chunk": chunk or {"kind": "fixed_char", "chunk_size": 800},
            "retrieve": retrieve or {"kind": "dense", "top_k": 10},
            "rerank": rerank or {"kind": "noop"},
        },
        "corpus": {"sha256": corpus_sha},
        "labels": {"sha256": labels_sha, "author": author},
        "per_query": rows,
    }


def metrics(recall: float) -> dict[str, float]:
    return {
        "recall@5": recall,
        "doc_recall@5": recall,
        "ndcg@5": recall,
        "mrr": recall,
        "chars@5": 4000.0,
    }


def write(tmp_path: Path, name: str, payload: dict[str, Any]) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# naming
# --------------------------------------------------------------------------------------


def test_the_name_carries_chunk_retrieve_and_rerank() -> None:
    name = compare.name_of(Path("x.json"), artifact(scored={"q1": metrics(1.0)}))
    assert name == "fixed_char/800 dense+noop"


def test_two_runs_differing_only_in_fetch_k_get_different_names() -> None:
    """Otherwise the pool-width sweep prints as one repeated row."""
    narrow = artifact(
        scored={"q1": metrics(1.0)},
        retrieve={"kind": "dense", "top_k": 10, "fetch_k": 20},
        rerank={"kind": "jina_rerank"},
    )
    wide = artifact(
        scored={"q1": metrics(1.0)},
        retrieve={"kind": "dense", "top_k": 10, "fetch_k": 50},
        rerank={"kind": "jina_rerank"},
    )
    assert compare.name_of(Path("a"), narrow) != compare.name_of(Path("b"), wide)
    assert "20" in compare.name_of(Path("a"), narrow)


def test_a_run_without_fetch_k_does_not_grow_an_empty_slash() -> None:
    assert "/" not in compare.name_of(Path("x"), artifact(scored={"q1": metrics(1.0)})).split()[1]


def test_a_sentence_chunker_is_named_by_its_own_size_field() -> None:
    config = {"kind": "sentence", "max_chars": 800, "overlap_sentences": 1}
    name = compare.name_of(Path("x"), artifact(scored={"q1": metrics(1.0)}, chunk=config))
    assert name.startswith("sentence/800")


# --------------------------------------------------------------------------------------
# the common subset
# --------------------------------------------------------------------------------------


def test_only_scored_queries_count() -> None:
    a = artifact(scored={"q1": metrics(1.0)}, unreachable=("q2",))
    assert compare.scored_ids(a) == {"q1"}


def test_the_mean_is_taken_over_the_named_ids_only() -> None:
    a = artifact(scored={"q1": metrics(1.0), "q2": metrics(0.0)})
    assert compare.mean_over(a, {"q1"}, "recall@5") == 1.0


def test_an_empty_subset_has_no_mean_rather_than_a_zero() -> None:
    """Reporting 0.0 for "nothing to average" would read as a measured failure."""
    a = artifact(scored={"q1": metrics(1.0)})
    assert compare.mean_over(a, set(), "recall@5") is None


def test_runs_are_averaged_over_the_intersection(tmp_path: Path) -> None:
    """The whole point: a query one run could not score must leave both denominators."""
    both = artifact(scored={"q1": metrics(1.0), "q2": metrics(0.0)})
    one = artifact(scored={"q1": metrics(1.0)}, unreachable=("q2",))

    text = compare.report([write(tmp_path, "a", both), write(tmp_path, "b", one)])

    assert "Averaged over the 1 queries" in text
    assert "1.0000" in text and "0.5000" not in text


def test_a_run_scoring_more_on_its_own_says_so(tmp_path: Path) -> None:
    both = artifact(scored={"q1": metrics(1.0), "q2": metrics(1.0)})
    one = artifact(scored={"q1": metrics(1.0)}, unreachable=("q2",))

    text = compare.report([write(tmp_path, "a", both), write(tmp_path, "b", one)])
    assert "(2 scored on its own)" in text


def test_dropped_queries_are_named(tmp_path: Path) -> None:
    both = artifact(scored={"q1": metrics(1.0), "q_boundary": metrics(1.0)})
    one = artifact(scored={"q1": metrics(1.0)}, unreachable=("q_boundary",))

    text = compare.report([write(tmp_path, "a", both), write(tmp_path, "b", one)])
    assert "q_boundary" in text


def test_nothing_is_flagged_when_every_run_scored_everything(tmp_path: Path) -> None:
    a = artifact(scored={"q1": metrics(1.0)})
    b = artifact(scored={"q1": metrics(0.5)})

    text = compare.report([write(tmp_path, "a", a), write(tmp_path, "b", b)])
    assert "Excluded" not in text and "scored on its own" not in text


# --------------------------------------------------------------------------------------
# incomparable runs
# --------------------------------------------------------------------------------------


def test_different_corpora_are_flagged(tmp_path: Path) -> None:
    a = artifact(scored={"q1": metrics(1.0)}, corpus_sha="one")
    b = artifact(scored={"q1": metrics(1.0)}, corpus_sha="two")

    text = compare.report([write(tmp_path, "a", a), write(tmp_path, "b", b)])
    assert "different corpora" in text


def test_different_label_sets_are_flagged(tmp_path: Path) -> None:
    """0.90 against one answer key and 0.60 against another is not a 30-point regression."""
    a = artifact(scored={"q1": metrics(1.0)}, labels_sha="one")
    b = artifact(scored={"q1": metrics(1.0)}, labels_sha="two")

    text = compare.report([write(tmp_path, "a", a), write(tmp_path, "b", b)])
    assert "different labels" in text


def test_every_label_author_is_named(tmp_path: Path) -> None:
    """Provenance travels with the number or the number means nothing."""
    a = artifact(scored={"q1": metrics(1.0)}, labels_sha="one", author="abhi")
    b = artifact(scored={"q1": metrics(1.0)}, labels_sha="two", author="a model")

    text = compare.report([write(tmp_path, "a", a), write(tmp_path, "b", b)])
    assert "Labels by: abhi" in text and "Labels by: a model" in text


# --------------------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------------------


def test_one_artifact_is_not_a_comparison(tmp_path: Path, monkeypatch, capsys) -> None:
    path = write(tmp_path, "a", artifact(scored={"q1": metrics(1.0)}))
    monkeypatch.setattr("sys.argv", ["compare.py", str(path)])

    assert compare.main() == 2
    assert "at least two" in capsys.readouterr().out


def test_missing_paths_are_skipped_rather_than_crashing(tmp_path: Path, monkeypatch) -> None:
    """A glob that matched nothing should not look like a comparison of one run."""
    monkeypatch.setattr("sys.argv", ["compare.py", str(tmp_path / "nope.json")])
    assert compare.main() == 2


def test_the_real_artifacts_still_compare() -> None:
    """A schema change in run.py that broke this would otherwise surface at report time."""
    results = sorted((compare.REPO_ROOT / "evals" / "results").glob("*.json"))
    if len(results) < 2:
        pytest.skip("fewer than two artifacts committed")
    assert "Averaged over" in compare.report(results[:2])
