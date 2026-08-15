"""Check a label file against the corpus.

    python evals/validate_labels.py evals/labels/retrieval.json

Verifies every gold span is a verbatim, unambiguous quote from the document it names, that
the corpus has not moved under the labels, and reports how many chunks each query would
have to find under the production chunking config.

Reads only. Nothing here writes to evals/labels/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT))

from app.rag.chunk.fixed_char import chunk_fixed_char  # noqa: E402
from app.rag.config import FixedCharChunkConfig  # noqa: E402
from app.rag.ingest.local import load_directory  # noqa: E402
from evals import label_schema  # noqa: E402

CORPUS_DIR = REPO_ROOT / "evals" / "corpus"
MANIFEST = REPO_ROOT / "evals" / "manifest.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    args = parser.parse_args()

    if not args.labels.exists():
        print(f"no such file: {args.labels}")
        return 2

    labels = label_schema.load(args.labels)
    documents = load_directory(CORPUS_DIR)
    doc_ids = {doc.label: doc.doc_id for doc in documents}

    problems = label_schema.check(labels, documents, label_schema.corpus_hash(MANIFEST))
    for problem in problems:
        print(f"  ERROR  {problem}")
    for note in label_schema.warnings(labels):
        print(f"  WARN   {note}")

    config = FixedCharChunkConfig(chunk_size=args.chunk_size, overlap=args.overlap)
    chunks = [chunk for doc in documents for chunk in chunk_fixed_char(doc, config)]

    print(
        f"\n{len(labels.queries)} queries "
        f"({len(labels.answerable)} answerable, {len(labels.unanswerable)} unanswerable) "
        f"over {len(documents)} documents / {len(chunks)} chunks "
        f"at {args.chunk_size}/{args.overlap}\n"
    )

    unreachable = 0
    for query in labels.answerable:
        matched = label_schema.relevant_chunks(chunks, query, doc_ids)
        flag = "  <-- no chunk contains any gold span" if not matched else ""
        if not matched:
            unreachable += 1
        print(f"  {query.id}  {len(matched):>2} relevant chunk(s)  {query.query[:52]}{flag}")

    if unreachable:
        print(
            f"\n{unreachable} answerable quer(y/ies) can never be satisfied at this chunk "
            "size: the span straddles a boundary. Shorten the span."
        )

    print()
    if problems:
        print(f"{len(problems)} error(s) — these labels would produce wrong numbers.")
        return 1
    print("Labels are consistent with the corpus.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
