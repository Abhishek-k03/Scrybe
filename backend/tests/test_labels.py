"""The label schema, the relevance rule, and the guard that labels stay hand-authored."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.rag.types import Chunk, Document
from evals import label_schema
from evals.label_schema import GoldSpan, LabelledQuery, LabelSet, is_relevant, relevant_chunks

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = REPO_ROOT / "evals" / "labels.example.json"
LABELS_DIR = REPO_ROOT / "evals" / "labels"
CORPUS_DIR = REPO_ROOT / "evals" / "corpus"
MANIFEST = REPO_ROOT / "evals" / "manifest.json"

HASH = "a" * 64


def query(**overrides) -> LabelledQuery:
    base = {
        "id": "q001",
        "query": "who created python",
        "gold": [GoldSpan(doc="guido.txt", span="creator of the Python programming language")],
    }
    return LabelledQuery(**{**base, **overrides})


def make_chunk(doc_id: str, text: str, index: int = 0) -> Chunk:
    return Chunk(
        doc_id=doc_id,
        doc_label=f"{doc_id}.txt",
        source_type="corpus",
        chunk_index=index,
        text=text,
        start_char=0,
        end_char=len(text),
    )


# --------------------------------------------------------------------------------------
# schema
# --------------------------------------------------------------------------------------


def test_an_answerable_query_needs_at_least_one_span() -> None:
    with pytest.raises(ValidationError, match="answerable query has no gold spans"):
        query(gold=[])


def test_an_unanswerable_query_must_have_no_spans() -> None:
    """Otherwise a half-finished label reads as a deliberate abstention case."""
    with pytest.raises(ValidationError, match="must have no gold spans"):
        query(answerable=False)


def test_an_unanswerable_query_is_valid_with_no_spans() -> None:
    assert LabelledQuery(id="q9", query="salary in berlin", answerable=False).gold == ()


def test_duplicate_query_ids_are_rejected() -> None:
    with pytest.raises(ValidationError, match=r"duplicate query ids: \['q001'\]"):
        LabelSet(corpus_sha256=HASH, queries=[query(), query()])


def test_a_mistyped_key_is_rejected_rather_than_ignored() -> None:
    with pytest.raises(ValidationError, match="gold_spans"):
        LabelledQuery(id="q1", query="x", gold_spans=[])


def test_an_empty_span_is_rejected() -> None:
    with pytest.raises(ValidationError):
        GoldSpan(doc="a.txt", span="")


def test_the_corpus_hash_must_be_a_full_sha256() -> None:
    with pytest.raises(ValidationError):
        LabelSet(corpus_sha256="abc123", queries=[query()])


def test_answerable_and_unanswerable_split() -> None:
    labels = LabelSet(
        corpus_sha256=HASH,
        queries=[query(), LabelledQuery(id="q2", query="nope", answerable=False)],
    )
    assert [q.id for q in labels.answerable] == ["q001"]
    assert [q.id for q in labels.unanswerable] == ["q2"]


# --------------------------------------------------------------------------------------
# relevance
# --------------------------------------------------------------------------------------


DOC_IDS = {"guido.txt": "aaa", "other.txt": "bbb"}
LABELLED = LabelledQuery(
    id="q1",
    query="who created python",
    gold=[GoldSpan(doc="guido.txt", span="creator of the Python")],
)


def test_a_chunk_containing_the_span_is_relevant() -> None:
    assert is_relevant(make_chunk("aaa", "He is the creator of the Python language"), LABELLED, DOC_IDS)


def test_the_right_text_in_the_wrong_document_is_not_relevant() -> None:
    """Both halves of the key have to match, or a quoting page would count as gold."""
    assert not is_relevant(make_chunk("bbb", "the creator of the Python"), LABELLED, DOC_IDS)


def test_the_right_document_without_the_span_is_not_relevant() -> None:
    assert not is_relevant(make_chunk("aaa", "Van Rossum was born in the Netherlands"), LABELLED, DOC_IDS)


def test_a_span_split_across_chunks_matches_neither() -> None:
    """Recall is 0 for such a query no matter how good retrieval is; the validator warns."""
    left = make_chunk("aaa", "He is the creator", 0)
    right = make_chunk("aaa", "of the Python language", 1)
    assert relevant_chunks([left, right], LABELLED, DOC_IDS) == []


def test_overlap_can_make_two_chunks_relevant() -> None:
    text = "He is the creator of the Python language"
    matched = relevant_chunks([make_chunk("aaa", text, 0), make_chunk("aaa", text, 1)], LABELLED, DOC_IDS)
    assert len(matched) == 2


def test_an_unanswerable_query_has_no_relevant_chunks() -> None:
    unanswerable = LabelledQuery(id="q9", query="salary", answerable=False)
    assert not is_relevant(make_chunk("aaa", "anything at all"), unanswerable, DOC_IDS)


def test_a_document_missing_from_the_corpus_never_matches() -> None:
    orphan = LabelledQuery(
        id="q1", query="x", gold=[GoldSpan(doc="absent.txt", span="text")]
    )
    assert not is_relevant(make_chunk("aaa", "text"), orphan, DOC_IDS)


# --------------------------------------------------------------------------------------
# checking against the corpus
# --------------------------------------------------------------------------------------


DOCS = [Document.create("guido.txt", "He is the creator of the Python programming language.")]
DOCS_BY_ID = {doc.label: doc.doc_id for doc in DOCS}


def check(labels: LabelSet, corpus_hash: str = HASH) -> list[str]:
    return label_schema.check(labels, DOCS, corpus_hash)


def test_a_clean_label_set_reports_nothing() -> None:
    labels = LabelSet(
        corpus_sha256=HASH,
        queries=[query(gold=[GoldSpan(doc="guido.txt", span="creator of the Python")])],
    )
    assert check(labels) == []


def test_a_moved_corpus_is_caught() -> None:
    """Ids are content hashes, so a changed corpus silently repoints every label."""
    labels = LabelSet(corpus_sha256=HASH, queries=[query(gold=[GoldSpan(doc="guido.txt", span="creator")])])
    assert any("corpus hash mismatch" in problem for problem in check(labels, "b" * 64))


def test_a_span_that_is_not_a_verbatim_quote_is_caught() -> None:
    labels = LabelSet(
        corpus_sha256=HASH,
        queries=[query(gold=[GoldSpan(doc="guido.txt", span="creator of Python")])],
    )
    assert any("not found verbatim" in problem for problem in check(labels))


def test_an_unknown_document_is_caught() -> None:
    labels = LabelSet(
        corpus_sha256=HASH,
        queries=[query(gold=[GoldSpan(doc="nope.txt", span="anything")])],
    )
    assert any("no corpus document named" in problem for problem in check(labels))


def test_an_ambiguous_span_is_caught() -> None:
    docs = [Document.create("d.txt", "Python is good. Python is good.")]
    labels = LabelSet(
        corpus_sha256=HASH,
        queries=[query(gold=[GoldSpan(doc="d.txt", span="Python is good.")])],
    )
    problems = label_schema.check(labels, docs, HASH)
    assert any("appears 2x" in problem for problem in problems)


def test_an_overlong_span_is_flagged() -> None:
    long_span = "x" * (label_schema.LONG_SPAN_CHARS + 1)
    docs = [Document.create("d.txt", long_span)]
    labels = LabelSet(corpus_sha256=HASH, queries=[query(gold=[GoldSpan(doc="d.txt", span=long_span)])])

    assert any("straddling a chunk boundary" in problem for problem in label_schema.check(labels, docs, HASH))


def test_a_label_set_with_no_unanswerable_queries_is_warned_about() -> None:
    labels = LabelSet(corpus_sha256=HASH, queries=[query()])
    assert any("abstention cannot be measured" in note for note in label_schema.warnings(labels))


# --------------------------------------------------------------------------------------
# the shipped example
# --------------------------------------------------------------------------------------


def test_the_example_parses() -> None:
    assert len(label_schema.load(EXAMPLE).queries) == 4


def test_the_example_pins_the_committed_corpus() -> None:
    assert label_schema.load(EXAMPLE).corpus_sha256 == label_schema.corpus_hash(MANIFEST)


def test_every_example_span_is_a_real_quote_from_the_corpus() -> None:
    """The examples are only useful as a template if they actually validate."""
    from app.rag.ingest.local import load_directory

    documents = load_directory(CORPUS_DIR)
    problems = label_schema.check(
        label_schema.load(EXAMPLE), documents, label_schema.corpus_hash(MANIFEST)
    )
    assert problems == []


def test_every_answerable_example_is_reachable_at_the_production_chunk_size() -> None:
    from app.rag.chunk.fixed_char import chunk_fixed_char
    from app.rag.config import FixedCharChunkConfig
    from app.rag.ingest.local import load_directory

    documents = load_directory(CORPUS_DIR)
    doc_ids = {doc.label: doc.doc_id for doc in documents}
    config = FixedCharChunkConfig()
    chunks = [chunk for doc in documents for chunk in chunk_fixed_char(doc, config)]

    for labelled in label_schema.load(EXAMPLE).answerable:
        assert relevant_chunks(chunks, labelled, doc_ids), f"{labelled.id} matches no chunk"


def test_the_example_lives_outside_the_labels_directory() -> None:
    """Anything under evals/labels/ is human ground truth; the example is not."""
    assert LABELS_DIR not in EXAMPLE.parents


# --------------------------------------------------------------------------------------
# rule 4: nothing writes to evals/labels/
# --------------------------------------------------------------------------------------


WRITE_METHODS = frozenset({"write_text", "write_bytes", "unlink", "replace", "rename", "rmdir"})
SOURCE_DIRS = (REPO_ROOT / "evals", REPO_ROOT / "backend" / "app")


def _write_targets(tree: ast.AST) -> list[str]:
    """Expressions that some call in this module writes to."""
    targets: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name) and function.id == "open" and len(node.args) > 1:
            mode = node.args[1]
            if isinstance(mode, ast.Constant) and any(c in str(mode.value) for c in "wax+"):
                targets.append(ast.unparse(node.args[0]))
        elif isinstance(function, ast.Attribute) and function.attr in WRITE_METHODS:
            targets.append(ast.unparse(function.value))
    return targets


def _source_files() -> list[Path]:
    return sorted(path for directory in SOURCE_DIRS for path in directory.rglob("*.py"))


def test_there_are_source_files_to_scan() -> None:
    assert _source_files(), "no modules found — this guard would pass vacuously"


def test_the_write_detector_actually_detects_writes() -> None:
    """An unverified guard gets trusted anyway."""
    tree = ast.parse(
        "labels_path.write_text('{}')\n"
        "open(LABELS_DIR / 'y.json', 'w')\n"
        "(root / 'results.json').write_text('{}')\n"
    )
    targets = _write_targets(tree)

    assert len(targets) == 3
    assert sum("label" in target.lower() for target in targets) == 2


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: p.name)
def test_no_module_writes_to_a_path_naming_labels(path: Path) -> None:
    """Labels are authored by hand. A script that rewrites them would fabricate truth.

    Static and therefore partial: it catches a write whose target expression names labels,
    not one laundered through an opaque variable. The session fingerprint in conftest
    covers what this cannot.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = [target for target in _write_targets(tree) if "label" in target.lower()]
    assert not offenders, f"{path.name} writes to {offenders}"


def test_the_validator_only_reads() -> None:
    source = (REPO_ROOT / "evals" / "validate_labels.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    assert _write_targets(tree) == []


def test_labels_directory_is_untracked_or_hand_written() -> None:
    """If the directory exists, every file in it must be valid JSON someone wrote."""
    if not LABELS_DIR.is_dir():
        pytest.skip("no labels authored yet")
    for path in sorted(LABELS_DIR.glob("*.json")):
        json.loads(path.read_text(encoding="utf-8"))
