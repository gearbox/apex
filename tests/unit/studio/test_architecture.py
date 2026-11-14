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


# -- S1-P1b: the verification greps of the importer prompt, as durable tests ---------------------


def test_the_plan_schema_is_v2_and_lyrics_stay_v1() -> None:
    from src.core.enums import StudioDocumentKind
    from src.studio.documents import CURRENT_SCHEMA_VERSION

    assert CURRENT_SCHEMA_VERSION[StudioDocumentKind.PLAN] == 2
    assert CURRENT_SCHEMA_VERSION[StudioDocumentKind.LYRICS] == 1


def test_migration_056_never_imports_application_code() -> None:
    source = (ROOT / "alembic/versions/056_studio_plan_schema_v2.py").read_text()
    assert not re.search(r"^\s*(from|import)\s+src\b", source, flags=re.MULTILINE)


def test_the_section_vocabulary_lives_only_in_section_names() -> None:
    for path in SRC.glob("studio/**/*.py"):
        assert not re.search(r"_display_name|_section_labels", path.read_text()), path
    text = (SRC / "studio/section_names.py").read_text()
    for name in ("is_header_text", "resolve_kind", "default_labels"):
        assert len(re.findall(rf"^def {name}\b", text, flags=re.MULTILINE)) == 1, name


def test_the_header_is_written_by_one_function_the_importer_reuses() -> None:
    suno = (SRC / "studio/renderers/suno.py").read_text()
    build = (SRC / "studio/importers/suno_build.py").read_text()
    assert "def render_header" in suno
    assert "render_header" in build
    assert "_meta_tag" not in build


def test_the_version_service_has_no_hard_coded_origin() -> None:
    text = (SRC / "studio/services/version.py").read_text()
    assert "StudioDocumentOrigin.USER" not in text
    assert "origin=origin.value" in text


def test_each_caller_states_its_origin() -> None:
    assert "origin=StudioDocumentOrigin.USER" in (SRC / "api/routes/studio_tracks.py").read_text()
    importer = (SRC / "studio/services/importer.py").read_text()
    assert importer.count("origin=StudioDocumentOrigin.IMPORT") == 2  # plan and lyrics
    assert "StudioDocumentOrigin.USER" not in importer


def test_the_importer_never_catches_broadly() -> None:
    """Catching a ``StudioError`` would let a half-applied import commit."""
    text = (SRC / "studio/services/importer.py").read_text()
    assert not re.search(r"except\s+(StudioError|Exception|BaseException)", text)
    assert not re.search(r"except\s*:", text)
    assert re.findall(r"except\s+(\w+)", text).count("VersionUnchanged") == 2


def test_the_parser_and_builder_are_pure() -> None:
    forbidden = {"sqlalchemy", "litestar", "structlog", "logging", "random", "time", "datetime"}
    for name in ("suno.py", "suno_build.py", "subtraction.py", "roundtrip.py", "__init__.py"):
        imported = _imported_modules(SRC / "studio/importers" / name)
        offending = {m for m in imported if m.split(".")[0] in forbidden or m.startswith("src.db")}
        assert not offending, (name, offending)


def _code_string_literals(path: Path) -> list[str]:
    """Every string constant of a module that is not a docstring (a bare string statement)."""
    tree = ast.parse(path.read_text())
    docstrings = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


def test_the_tempo_is_defined_once_and_nobody_else_spells_it() -> None:
    """R1/R5: the renderer writes it and the importer reads it through ``src/studio/tempo.py``."""
    text = (SRC / "studio/tempo.py").read_text()
    for name in ("tempo_of", "tempo_token"):
        assert len(re.findall(rf"^def {name}\b", text, flags=re.MULTILINE)) == 1, name
    for path in SRC.glob("studio/**/*.py"):
        if path.name == "tempo.py":
            continue
        assert not re.search(
            r"def tempo_of|def tempo_token|_bpm_of|_MIN_BPM|_MAX_BPM", path.read_text()
        )
    for name in ("renderers/suno.py", "importers/suno.py"):
        literals = _code_string_literals(SRC / "studio" / name)
        assert not [lit for lit in literals if re.search(r"bpm", lit, flags=re.IGNORECASE)], name
    # the only regex that mentions "bpm" is the module's own
    for path in SRC.glob("studio/**/*.py"):
        if path.name != "tempo.py":
            assert not re.search(r"re\.compile\([^\n]*bpm", path.read_text(), flags=re.IGNORECASE)


def test_the_importer_and_the_validators_use_the_shared_tempo_module() -> None:
    assert "tempo_of" in (SRC / "studio/importers/suno.py").read_text()
    assert "tempo_of" in (SRC / "studio/validation.py").read_text()
    assert "tempo_token" in (SRC / "studio/renderers/suno.py").read_text()
    assert "from src.studio.tempo import" in (SRC / "studio/documents.py").read_text()


def test_the_contract_documents_what_p1b_added() -> None:
    contract = (ROOT / "docs/contracts/studio-contract.md").read_text()
    for needle in (
        "reserved_descriptor",
        "reserved_delivery_tag",
        "reserved_tempo_tag",
        "instrumental_plan_has_lines",
        "import_invalid",
        "unsupported_inline_tag",
        "instrumental_with_lyric_lines",
        "language_required",
        "round_trip",
    ):
        assert needle in contract, needle
    assert len(re.findall(r"^## 10\. Suno import", contract, flags=re.MULTILINE)) == 1
    assert "resets the flag" not in contract


def test_hypothesis_is_a_pinned_dev_dependency() -> None:
    assert re.search(r'"hypothesis>=6', (ROOT / "pyproject.toml").read_text())
    assert re.search(r'^name = "hypothesis"', (ROOT / "uv.lock").read_text(), flags=re.MULTILINE)
