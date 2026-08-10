"""Content-addressed disk cache.

Keys are hashes of everything that affects the result, so a cached value can never be
returned for a different model, task, or input. Re-running a sweep must not re-pay for
API calls, and it must not silently reuse a value from a different configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def content_key(*parts: str) -> str:
    """Hash of the parts, NUL-separated so ('ab','c') and ('a','bc') differ."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


class DiskCache:
    """One JSON file per key, sharded by the first two hex characters."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.hits = 0
        self.misses = 0

    def path_for(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> Any | None:
        path = self.path_for(key)
        if not path.exists():
            self.misses += 1
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A truncated file from an interrupted write is a miss, not a crash.
            self.misses += 1
            return None
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file and rename so a crash cannot leave a partial entry.
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    def reset_stats(self) -> None:
        self.hits = 0
        self.misses = 0
