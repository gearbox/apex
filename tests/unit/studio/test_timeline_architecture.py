"""The structural rules of realized timelines (S3a), as durable tests."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"


def _imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def _transitive_src_imports(module: str, seen: set[str]) -> set[str]:
    """Every ``src`` module ``module`` pulls in at import time, directly or not.

    ``TYPE_CHECKING`` imports are included, which only makes the rule stricter.
    """
    if module in seen:
        return seen
    seen.add(module)
    relative = Path(*module.split("."))
    for candidate in (ROOT / relative.with_suffix(".py"), ROOT / relative / "__init__.py"):
        if candidate.is_file():
            for imported in _imported_modules(candidate):
                if imported == "src" or imported.startswith("src."):
                    _transitive_src_imports(imported, seen)
    return seen


def test_the_timeline_module_imports_nothing_from_the_api_or_the_database() -> None:
    reached = _transitive_src_imports("src.studio.timeline", set())

    assert "src.core.media_limits" in reached  # the walk really follows imports
    assert not {m for m in reached if m.startswith(("src.api", "src.db"))}, sorted(reached)


def test_the_timeline_module_does_no_io() -> None:
    forbidden = {"sqlalchemy", "litestar", "structlog", "logging", "httpx", "asyncio"}
    imported = _imported_modules(SRC / "studio/timeline.py")

    assert not {m for m in imported if m.split(".")[0] in forbidden}


def test_the_audio_duration_cap_is_defined_once() -> None:
    """Media ingest and the timeline bounds share ``src/core/media_limits.py`` (S3-3)."""
    definers = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if re.search(r"^AUDIO_MAX_SECONDS\b\s*[:=]", path.read_text(), flags=re.MULTILINE)
    ]

    assert definers == ["core/media_limits.py"]
    for name in ("api/services/media_ingest/audio.py", "studio/timeline.py"):
        assert "src.core.media_limits" in _imported_modules(SRC / name), name
