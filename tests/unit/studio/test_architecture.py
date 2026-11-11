"""The prompt's verification greps, as durable tests (invariants 5 and the structural rules)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"

LIBRARY_FILES = sorted(
    {
        *SRC.glob("api/services/library*.py"),
        *SRC.glob("api/routes/library*.py"),
        *SRC.glob("db/repositories/library*.py"),
        SRC / "api/schemas/library.py",
        SRC / "db/models/library.py",
        SRC / "core/library_ref.py",
        SRC / "core/library_limits.py",
    }
)
STUDIO_FILES = sorted(
    {
        *SRC.glob("studio/**/*.py"),
        SRC / "db/models/studio.py",
        SRC / "db/repositories/studio.py",
        *SRC.glob("api/routes/studio_*.py"),
        SRC / "api/schemas/studio.py",
        SRC / "api/security/studio.py",
    }
)


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text())
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module)
    return modules


def test_the_file_sets_are_not_empty() -> None:
    assert len(LIBRARY_FILES) >= 8
    assert len(STUDIO_FILES) >= 20


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_the_library_never_imports_the_studio(path: Path) -> None:
    offending = {
        m for m in _imported_modules(path) if m == "src.studio" or m.startswith("src.studio.")
    }
    assert not offending, f"{path.relative_to(ROOT)} imports {offending}"


def test_the_library_defines_the_port_and_studio_implements_it() -> None:
    guard_module = SRC / "api/services/library_project_guards.py"
    assert not {m for m in _imported_modules(guard_module) if "studio" in m}
    studio_guard = SRC / "studio/services/project_deletion_guard.py"
    assert "src.api.services.library_project_guards" in _imported_modules(studio_guard)


def test_renderers_never_name_the_full_document_or_the_artist_row() -> None:
    for name in ("suno.py", "base.py"):
        text = (SRC / "studio/renderers" / name).read_text()
        assert not re.search(r"PlanDocument|StudioArtist\b", text), name
        imported = _imported_modules(SRC / "studio/renderers" / name)
        assert "src.db.models.studio" not in imported


def test_the_studio_repository_never_flushes_or_commits() -> None:
    text = (SRC / "db/repositories/studio.py").read_text()
    assert not re.search(r"session\.(commit|flush)\(", text)


@pytest.mark.parametrize("path", STUDIO_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_ids_come_from_new_id_only(path: Path) -> None:
    text = path.read_text()
    assert not re.search(r"uuid4|uuid_utils", text), path


@pytest.mark.parametrize("path", STUDIO_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_f_string_log_calls(path: Path) -> None:
    assert not re.search(r'logger\.\w+\(f"', path.read_text()), path


def test_versions_are_numbered_from_counters_not_max() -> None:
    for path in (SRC / "db/repositories/studio.py", *SRC.glob("studio/**/*.py")):
        assert not re.search(r"max\(.*version_number", path.read_text()), path


def test_every_studio_controller_runs_the_feature_guard_first() -> None:
    controllers = sorted(SRC.glob("api/routes/studio_*.py"))
    assert len(controllers) == 3
    for path in controllers:
        text = path.read_text()
        assert "guards = [studio_feature_guard, auth_guard]" in text, path
        assert "guards = [auth_guard" not in text, path


def test_no_studio_route_is_added_to_the_legal_exempt_opt() -> None:
    for path in SRC.glob("api/routes/studio_*.py"):
        assert "legal_exempt" not in path.read_text(), path
