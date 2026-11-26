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


def _contract_section_16() -> str:
    contract = (ROOT / "docs/contracts/studio-contract.md").read_text()
    assert len(re.findall(r"^## 16\. Timelines", contract, flags=re.MULTILINE)) == 1
    return contract[contract.index("\n## 16. Timelines") :]


def _issue_codes() -> list[str]:
    """Every validation code the timeline module declares (its ``Final`` code constants)."""
    from src.studio import timeline

    codes = [
        value
        for name, value in vars(timeline).items()
        if name.isupper() and isinstance(value, str) and name == value.upper()
    ]
    assert len(codes) == 14  # S3-4
    return codes


def test_the_contract_documents_every_validation_code_error_and_warning() -> None:
    from src.studio import errors
    from src.studio.renderers import timed_lyrics

    section = _contract_section_16()
    needles = [
        *_issue_codes(),
        *(
            error.code
            for error in (
                errors.TimelineNotFound,
                errors.AudioHasNoPlan,
                errors.TimelineUnchanged,
                errors.TimelineLimitReached,
                errors.TimelineIsCurrent,
                errors.TimelineHasChildren,
                errors.AudioHasTimelines,
            )
        ),
        timed_lyrics.UNTIMED_LINES,
        timed_lyrics.NO_LYRICS,
        timed_lyrics.INFERRED_WORD_TIMINGS,
        "too_long_after_normalization",
        "document_too_large",
    ]
    missing = [needle for needle in needles if not re.search(rf"\b{needle}\b", section)]

    assert not missing, missing
    assert "never invents" in section
    assert "invalid_peaks_window" in (ROOT / "docs/BACKEND_API_REFERENCE.md").read_text()
    assert "peaks_not_ready" in (ROOT / "docs/BACKEND_API_REFERENCE.md").read_text()


def test_the_conventions_record_the_timelines() -> None:
    text = (ROOT / "CLAUDE.md").read_text()
    for needle in (
        "**Timelines (S3a, migration 063",
        "backfill_audio_peaks",
        "StudioAudioTimeline",
        "detail_peaks",
        "studio_timelines.py",
        "timed_lyrics.py",
    ):
        assert needle in text, needle
