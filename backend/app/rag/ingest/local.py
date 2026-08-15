"""Load a directory of text files as documents.

The eval corpus lives on disk; loading it must produce the same documents in the same order
on every machine, so paths are sorted and ids come from content.
"""

from __future__ import annotations

from pathlib import Path

from app.rag.types import Document


def load_file(path: str | Path, source_type: str = "corpus") -> Document:
    resolved = Path(path)
    return Document.create(
        label=resolved.name,
        text=resolved.read_text(encoding="utf-8"),
        source_type=source_type,
    )


def load_directory(
    directory: str | Path,
    pattern: str = "*.txt",
    source_type: str = "corpus",
) -> list[Document]:
    root = Path(directory)
    if not root.is_dir():
        raise NotADirectoryError(f"corpus directory not found: {root}")
    return [load_file(path, source_type) for path in sorted(root.glob(pattern))]
