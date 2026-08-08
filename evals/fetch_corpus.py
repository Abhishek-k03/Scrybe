"""Fetch the evaluation corpus from Wikipedia as plain text.

Run:  python evals/fetch_corpus.py [--force]

Every article is pinned to the exact revision id it was fetched at, and every file is
recorded with its sha256 in `manifest.json`. This matters more than it looks: Wikipedia
articles change continuously, so a corpus identified only by title is not a fixed corpus,
and a recall@k measured against it last week is not comparable to one measured today.

The manifest is what makes an eval artifact honest — `evals/run.py` records the corpus
hash alongside its metrics so any result can be traced to the exact bytes that produced it.

This script only writes to evals/corpus/. It never touches backend/chroma_db/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

API = "https://en.wikipedia.org/w/api.php"
CORPUS_DIR = Path(__file__).parent / "corpus"
MANIFEST = Path(__file__).parent / "manifest.json"

# Chosen for factual density and topical spread across the Python ecosystem: the language
# itself, its implementations, packaging, the scientific stack, web frameworks, and the
# language concepts that Python questions tend to be about. Distinct enough that a query
# has a genuinely correct source rather than four plausible ones.
TITLES = [
    # language and implementations
    "Python (programming language)",
    "CPython",
    "PyPy",
    "Jython",
    "IronPython",
    "MicroPython",
    "Cython",
    "Global interpreter lock",
    "Python syntax and semantics",
    # people and governance
    "Guido van Rossum",
    "Python Software Foundation",
    # packaging
    "Python Package Index",
    "Pip (package manager)",
    "Anaconda (Python distribution)",
    # scientific stack
    "NumPy",
    "Pandas (software)",
    "SciPy",
    "Matplotlib",
    "Scikit-learn",
    "PyTorch",
    "SymPy",
    # web and application frameworks
    "Django (web framework)",
    "Flask (web framework)",
    "Tornado (web server)",
    "SQLAlchemy",
    "Twisted (software)",
    "Beautiful Soup (HTML parser)",
    # tooling and interactive computing
    "Project Jupyter",
    "IPython",
    "Tkinter",
    "PyGame",
    # language concepts Python questions lean on
    "List comprehension",
    "Duck typing",
    "Generator (computer programming)",
    "Reference counting",
]


def slugify(title: str) -> str:
    keep = [c.lower() if c.isalnum() else "_" for c in title]
    slug = "".join(keep)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def fetch(client: httpx.Client, title: str) -> tuple[str, str, int] | None:
    """Return (resolved_title, plain_text, revision_id), or None if the page is missing."""
    resp = client.get(
        API,
        params={
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|revisions",
            "rvprop": "ids",
            "explaintext": "1",
            "redirects": "1",
            "titles": title,
        },
    )
    resp.raise_for_status()
    pages = resp.json().get("query", {}).get("pages", [])
    if not pages:
        return None

    page = pages[0]
    if page.get("missing"):
        return None

    text = (page.get("extract") or "").strip()
    revisions = page.get("revisions") or []
    revid = revisions[0]["revid"] if revisions else 0
    if not text:
        return None
    return page["title"], text, revid


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-fetch and overwrite articles that are already present",
    )
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    existing = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {}
    by_title = {entry["title"]: entry for entry in existing.get("documents", [])}

    documents: list[dict[str, object]] = []
    skipped: list[str] = []

    # Wikimedia's User-Agent policy requires "Client/version (contact-url) library/version"
    # and returns 403 for generic agents. Do not simplify this string.
    # https://foundation.wikimedia.org/wiki/Policy:Wikimedia_Foundation_User-Agent_Policy
    headers = {
        "User-Agent": (
            "scrybe-evals/0.1 (https://github.com/scrybe/evals; retrieval-eval-corpus) "
            "python-httpx/0.28.1"
        )
    }
    with httpx.Client(timeout=30, headers=headers, follow_redirects=True) as client:
        for title in TITLES:
            slug = slugify(title)
            path = CORPUS_DIR / f"{slug}.txt"

            if path.exists() and not args.force and title in by_title:
                documents.append(by_title[title])
                print(f"  skip   {slug} (already present)")
                continue

            result = fetch(client, title)
            if result is None:
                skipped.append(title)
                print(f"  MISS   {title!r} — no such page", file=sys.stderr)
                continue

            resolved, text, revid = result
            path.write_text(text, encoding="utf-8")
            documents.append(
                {
                    "title": resolved,
                    "requested_title": title,
                    "slug": slug,
                    "file": f"corpus/{slug}.txt",
                    "revision_id": revid,
                    "url": f"https://en.wikipedia.org/w/index.php?oldid={revid}",
                    "chars": len(text),
                    "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                }
            )
            print(f"  ok     {slug:38s} {len(text):7,d} chars  rev {revid}")

    documents.sort(key=lambda d: str(d["slug"]))
    corpus_hash = hashlib.sha256(
        "".join(str(d["sha256"]) for d in documents).encode("utf-8")
    ).hexdigest()

    MANIFEST.write_text(
        json.dumps(
            {
                "source": "en.wikipedia.org",
                "license": "CC BY-SA 4.0",
                "document_count": len(documents),
                "total_chars": sum(int(d["chars"]) for d in documents),
                "corpus_sha256": corpus_hash,
                "documents": documents,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"\n{len(documents)} documents, {sum(int(d['chars']) for d in documents):,} chars")
    print(f"corpus_sha256 = {corpus_hash}")
    if skipped:
        print(f"skipped (not found): {skipped}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
