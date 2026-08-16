"""The eval harness: what it measures, and what it records about how it measured it."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.rag.config import (
    DenseRetrieveConfig,
    FakeEmbedConfig,
    FixedCharChunkConfig,
    JinaEmbedConfig,
    JinaRerankConfig,
    MemoryIndexConfig,
    NoopRerankConfig,
    PipelineConfig,
)
from app.rag.pipeline import build_pipeline
from app.rag.types import Document
from evals import run as harness
from evals.label_schema import GoldSpan, LabelledQuery, LabelSet

DOCS = [
    Document.create(
        "gil.txt",
        "The global interpreter lock means only one thread may execute bytecode at a time. "
        "This limits parallelism on multi-core machines for CPU bound work.",
    ),
    Document.create(
        "guido.txt",
        "Guido van Rossum created Python in 1989 and released it in 1991. "
        "He stepped down as benevolent dictator for life in 2018.",
    ),
]

LABELS = LabelSet(
    author="tester",
    corpus_sha256="c" * 64,
    queries=[
        LabelledQuery(
            id="q1",
            query="who made python",
            gold=[GoldSpan(doc="guido.txt", span="Guido van Rossum created Python in 1989")],
        ),
        LabelledQuery(
            id="q2",
            query="why is threading limited",
            gold=[GoldSpan(doc="gil.txt", span="only one thread may execute bytecode at a time")],
        ),
        LabelledQuery(id="u1", query="what is the price of tea", answerable=False),
    ],
)


def offline_config(**overrides) -> PipelineConfig:
    base = {
        "chunk": FixedCharChunkConfig(chunk_size=400, overlap=50),
        "embed": FakeEmbedConfig(dimensions=256),
        "index": MemoryIndexConfig(),
        "retrieve": DenseRetrieveConfig(top_k=3),
        "rerank": NoopRerankConfig(),
    }
    return PipelineConfig(**{**base, **overrides})


@pytest.fixture
async def evaluated():
    pipeline = build_pipeline(offline_config())
    report = await pipeline.index_documents(DOCS)
    doc_ids = {doc.label: doc.doc_id for doc in DOCS}
    outcomes = await harness.evaluate(pipeline, LABELS, doc_ids, top_k=3)
    return outcomes, report


# --------------------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------------------


async def test_every_labelled_query_is_evaluated(evaluated) -> None:
    outcomes, _ = evaluated
    assert [outcome.query_id for outcome in outcomes] == ["q1", "q2", "u1"]


async def test_relevant_sets_come_from_the_labels_not_the_retriever(evaluated) -> None:
    """The answer key must not depend on what was returned."""
    outcomes, _ = evaluated
    by_id = {outcome.query_id: outcome for outcome in outcomes}

    assert len(by_id["q1"].relevant_chunk_ids) == 1
    assert by_id["u1"].relevant_chunk_ids == frozenset()


async def test_an_unanswerable_query_has_no_relevant_chunks_however_it_ranks(evaluated) -> None:
    outcomes, _ = evaluated
    unanswerable = next(o for o in outcomes if o.query_id == "u1")

    assert unanswerable.relevant_chunk_ids == frozenset()
    assert not unanswerable.answerable


async def test_retrieval_without_a_threshold_never_abstains(evaluated) -> None:
    """Audit item 4, measured rather than asserted."""
    outcomes, _ = evaluated
    assert not any(outcome.abstained for outcome in outcomes)


async def test_a_threshold_makes_abstention_possible() -> None:
    """The same corpus and labels, with a score floor that nothing clears."""
    config = offline_config(retrieve=DenseRetrieveConfig(top_k=3, score_threshold=0.99))
    pipeline = build_pipeline(config)
    await pipeline.index_documents(DOCS)

    outcomes = await harness.evaluate(
        pipeline, LABELS, {doc.label: doc.doc_id for doc in DOCS}, top_k=3
    )
    assert all(outcome.abstained for outcome in outcomes)


async def test_doc_ids_are_recorded_alongside_chunk_ids(evaluated) -> None:
    outcomes, _ = evaluated
    for outcome in outcomes:
        assert len(outcome.retrieved_doc_ids) == len(outcome.retrieved_chunk_ids)


# --------------------------------------------------------------------------------------
# per-query records
# --------------------------------------------------------------------------------------


async def test_every_query_gets_a_traceable_row(evaluated) -> None:
    outcomes, _ = evaluated
    rows = harness.per_query_records(outcomes, LABELS)

    assert len(rows) == len(LABELS.queries)
    for row in rows:
        assert {"id", "query", "answerable", "n_relevant", "retrieved"} <= set(row)


async def test_answerable_rows_carry_their_own_scores(evaluated) -> None:
    """Per-query scores are what let two runs be re-averaged over a common subset."""
    outcomes, _ = evaluated
    rows = {row["id"]: row for row in harness.per_query_records(outcomes, LABELS)}

    assert {"mrr", "recall@5", "ndcg@5", "chars@5"} <= set(rows["q1"]["metrics"])
    assert rows["q1"]["scored"] is True


async def test_unanswerable_rows_are_not_scored(evaluated) -> None:
    outcomes, _ = evaluated
    rows = {row["id"]: row for row in harness.per_query_records(outcomes, LABELS)}

    assert rows["u1"]["scored"] is False
    assert "metrics" not in rows["u1"]
    assert "abstained" in rows["u1"]


async def test_the_context_budget_is_recorded_per_query(evaluated) -> None:
    """recall@k rises with chunk size for free unless its cost is measured beside it."""
    outcomes, _ = evaluated
    rows = {row["id"]: row for row in harness.per_query_records(outcomes, LABELS)}

    row = rows["q1"]
    assert row["metrics"]["chars@1"] <= row["metrics"]["chars@5"]
    assert sum(row["retrieved_chunk_chars"]) == row["metrics"]["chars@10"]


async def test_a_query_with_no_reachable_chunk_is_marked_not_scored() -> None:
    """A gold span split across chunks would otherwise be scored as a retrieval failure."""
    unreachable = LabelSet(
        author="tester",
        corpus_sha256="c" * 64,
        queries=[
            LabelledQuery(
                id="q1",
                query="who made python",
                gold=[GoldSpan(doc="guido.txt", span="a span that no chunk contains")],
            )
        ],
    )
    pipeline = build_pipeline(offline_config())
    await pipeline.index_documents(DOCS)
    outcomes = await harness.evaluate(
        pipeline, unreachable, {doc.label: doc.doc_id for doc in DOCS}, top_k=3
    )

    row = harness.per_query_records(outcomes, unreachable)[0]
    assert row["scored"] is False
    assert "metrics" not in row


# --------------------------------------------------------------------------------------
# the artifact — CLAUDE.md rule 2
# --------------------------------------------------------------------------------------


@pytest.fixture
async def artifact(evaluated):
    outcomes, report = evaluated
    return harness.build_artifact(
        offline_config(), LABELS, outcomes, len(DOCS), report.chunks_added, top_k=3
    )


async def test_the_artifact_records_the_full_config(artifact) -> None:
    """A number without its configuration cannot be reproduced or compared."""
    for stage in ("chunk", "embed", "index", "retrieve", "rerank"):
        assert stage in artifact["config"]
    assert artifact["config"]["chunk"]["chunk_size"] == 400


async def test_the_artifact_records_the_commit_and_whether_it_was_dirty(artifact) -> None:
    """A dirty tree means the SHA does not describe the code that ran."""
    assert set(artifact["git"]) == {"sha", "branch", "dirty", "n_untracked_files"}
    assert isinstance(artifact["git"]["dirty"], bool)


async def test_untracked_files_do_not_count_as_a_dirty_tree(artifact) -> None:
    """The run writes its own artifact; that must not mark its own SHA unreproducible."""
    assert isinstance(artifact["git"]["n_untracked_files"], int)


async def test_the_artifact_records_sample_sizes(artifact) -> None:
    assert artifact["corpus"]["n_docs"] == 2
    assert artifact["corpus"]["n_chunks"] > 0
    assert artifact["metrics"]["n_answerable"] == 2
    assert artifact["metrics"]["n_unanswerable"] == 1


async def test_the_artifact_pins_the_model(artifact) -> None:
    assert artifact["run"]["embed_model"] == "fake"

    jina = harness.build_artifact(
        offline_config(embed=JinaEmbedConfig()), LABELS, [], 0, 0, top_k=3
    )
    assert jina["run"]["embed_model"] == "jina-embeddings-v3"


async def test_the_artifact_records_the_label_provenance(artifact) -> None:
    assert artifact["labels"]["author"] == "tester"
    assert "sha256" in artifact["labels"]


async def test_a_model_authored_label_set_carries_its_caveat_into_the_artifact() -> None:
    labels = LABELS.model_copy(update={"author": "claude-opus-5"})
    built = harness.build_artifact(offline_config(), labels, [], 0, 0, top_k=3)

    assert any("not independent ground truth" in c for c in built["labels"]["caveats"])


async def test_the_artifact_is_json_serialisable(artifact) -> None:
    assert json.loads(json.dumps(artifact))["metrics"]["n_queries"] == 3


async def test_the_artifact_never_contains_the_api_key() -> None:
    built = harness.build_artifact(
        offline_config(embed=JinaEmbedConfig()), LABELS, [], 0, 0, top_k=3
    )
    assert "api_key" not in json.dumps(built).lower()


async def test_the_summary_renders_without_a_measurement(artifact) -> None:
    """Empty groups print n/a rather than formatting None as a number."""
    empty = harness.build_artifact(offline_config(), LABELS, [], 0, 0, top_k=3)
    assert "n/a" in harness.format_summary(empty)


# --------------------------------------------------------------------------------------
# path resolution
# --------------------------------------------------------------------------------------


def test_a_relative_cache_dir_resolves_against_the_repo_not_the_cwd() -> None:
    """Otherwise the cache lands wherever the script happened to be launched from."""
    config = offline_config(embed=JinaEmbedConfig(cache_dir=".cache/embeddings"))
    resolved = harness.resolve_paths(config).embed.cache_dir

    assert Path(resolved).is_absolute()
    assert Path(resolved) == harness.REPO_ROOT / ".cache" / "embeddings"


def test_an_absolute_cache_dir_is_left_alone(tmp_path: Path) -> None:
    config = offline_config(embed=JinaEmbedConfig(cache_dir=str(tmp_path)))
    assert harness.resolve_paths(config).embed.cache_dir == str(tmp_path)


def test_a_relative_rerank_cache_dir_also_resolves() -> None:
    """The reranker is the expensive call now; leaving it unresolved re-pays for the sweep."""
    config = offline_config(rerank=JinaRerankConfig(cache_dir=".cache/rerank"))
    resolved = harness.resolve_paths(config).rerank.cache_dir

    assert Path(resolved) == harness.REPO_ROOT / ".cache" / "rerank"


def test_both_caches_resolve_in_one_pass() -> None:
    config = offline_config(
        embed=JinaEmbedConfig(cache_dir=".cache/embeddings"),
        rerank=JinaRerankConfig(cache_dir=".cache/rerank"),
    )
    resolved = harness.resolve_paths(config)

    assert Path(resolved.embed.cache_dir).is_absolute()
    assert Path(resolved.rerank.cache_dir).is_absolute()


def test_a_config_without_a_cache_is_unchanged() -> None:
    config = offline_config()
    assert harness.resolve_paths(config) is config


def test_the_baseline_config_matches_what_the_app_runs() -> None:
    """The baseline must be today's production settings, or it is not a baseline."""
    from app.services.pipeline import default_config

    baseline = harness.baseline_config()
    production = default_config()

    assert baseline.chunk == production.chunk
    assert baseline.rerank.kind == production.rerank.kind == "noop"
    assert baseline.retrieve.kind == production.retrieve.kind == "dense"


def test_the_baseline_caches_embeddings() -> None:
    """CLAUDE.md rule 3: re-running a sweep must not re-pay for the API."""
    assert harness.baseline_config().embed.cache_dir is not None


def test_the_baseline_uses_exact_search() -> None:
    """ANN recall would vary between runs and be indistinguishable from a real change."""
    assert harness.baseline_config().index.kind == "memory"


# --------------------------------------------------------------------------------------
# --labels CLI flag
# --------------------------------------------------------------------------------------


async def test_a_relative_labels_path_does_not_crash(monkeypatch, tmp_path: Path) -> None:
    """A bare Path("evals/labels/x.json") has no anchor, so relative_to(REPO_ROOT) rejects
    it outright even when the file genuinely lives inside the repo. main() must resolve
    --labels against the CWD before comparing it to REPO_ROOT."""
    monkeypatch.chdir(harness.REPO_ROOT)
    monkeypatch.setattr(
        "sys.argv",
        [
            "run.py",
            "--offline",
            "--dry-run",
            "--labels",
            "evals/labels/retrieval.json",  # relative, exactly as a user would type it
        ],
    )
    assert await harness.main() == 0


async def test_scoring_against_a_different_label_file_uses_that_files_hash(tmp_path: Path) -> None:
    """The artifact's labels.sha256 must match whichever file was actually scored, not the
    default retrieval.json, or two artifacts would look interchangeable when they aren't."""
    alternate = tmp_path / "alt.json"
    alternate.write_text(LABELS.model_dump_json(), encoding="utf-8")

    pipeline = build_pipeline(offline_config())
    await pipeline.index_documents(DOCS)
    outcomes = await harness.evaluate(
        pipeline, LABELS, {doc.label: doc.doc_id for doc in DOCS}, top_k=3
    )
    built = harness.build_artifact(
        offline_config(), LABELS, outcomes, len(DOCS), 0, top_k=3, labels_path=alternate
    )

    assert built["labels"]["sha256"] == harness.file_sha256(alternate)


def test_repo_relative_path_is_shortened_inside_the_repo() -> None:
    inside = harness.REPO_ROOT / "evals" / "labels" / "retrieval.json"
    assert harness.repo_relative(inside) == "evals/labels/retrieval.json"


def test_repo_relative_path_falls_back_to_absolute_outside_the_repo(tmp_path: Path) -> None:
    """A scratch label file elsewhere on disk is a legitimate --labels target; recording
    its path must not raise just because it isn't under the repo."""
    outside = tmp_path / "scratch.json"
    assert harness.repo_relative(outside) == outside.as_posix()
