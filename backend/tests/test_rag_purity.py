"""`app.rag` must stay importable without the web app.

Enforced by reading the source rather than by importing, so a violation is caught even if
the forbidden module happens to be installed and imports cleanly.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

RAG_DIR = Path(__file__).resolve().parents[1] / "app" / "rag"

FORBIDDEN_PREFIXES = (
    "fastapi",
    "starlette",
    "app.core",
    "app.api",
    "app.services",
    "supabase",
    "groq",
)


def _module_files() -> list[Path]:
    return sorted(RAG_DIR.rglob("*.py"))


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.add(node.module)
    return found


def test_rag_package_exists() -> None:
    assert _module_files(), "no modules found — this guard would pass vacuously"


@pytest.mark.parametrize("path", _module_files(), ids=lambda p: p.name)
def test_module_imports_nothing_from_the_web_app(path: Path) -> None:
    offenders = {
        name
        for name in _imported_modules(path)
        for prefix in FORBIDDEN_PREFIXES
        if name == prefix or name.startswith(f"{prefix}.")
    }
    assert not offenders, f"{path.name} imports {sorted(offenders)}"


def test_importing_rag_does_not_pull_in_fastapi() -> None:
    """Import in a clean subprocess so an earlier test cannot mask the dependency."""
    import subprocess
    import sys

    backend = str(RAG_DIR.parents[1])
    code = (
        f"import sys; sys.path.insert(0, {backend!r});"
        "import app.rag;"
        "assert 'fastapi' not in sys.modules, 'fastapi was imported';"
        "assert 'app.services' not in sys.modules, 'app.services was imported';"
        "print('ok')"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_rag_does_not_read_environment_variables() -> None:
    """Config arrives as an argument; nothing here may reach for os.environ or dotenv."""
    offenders = []
    for path in _module_files():
        source = path.read_text(encoding="utf-8")
        if "os.environ" in source or "getenv" in source or "load_dotenv" in source:
            offenders.append(path.name)
    assert not offenders, f"modules reading the environment: {offenders}"
