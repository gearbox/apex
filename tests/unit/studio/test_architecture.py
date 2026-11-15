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
        SRC / "api/schemas/studio_runs.py",
        SRC / "api/security/studio.py",
        SRC / "workers/studio_run.py",
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
    assert len(controllers) == 4
    for path in controllers:
        text = path.read_text()
        if path.name == "studio_runs.py":  # the run switch sits between the feature guard and auth
            assert "guards = [studio_feature_guard, studio_runs_guard, auth_guard]" in text, path
        else:
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


# -- S1-P2a-A: the verification greps of the planner-runs prompt, as durable tests ---------------

PLANNER = SRC / "studio/planner"
EXECUTOR = SRC / "studio/services/run_executor.py"
RUN_SERVICE = SRC / "studio/services/run.py"


def test_langgraph_is_a_pinned_dependency() -> None:
    assert re.search(r'"langgraph>=1\.2,<2"', (ROOT / "pyproject.toml").read_text())
    assert re.search(r'^name = "langgraph"$', (ROOT / "uv.lock").read_text(), flags=re.MULTILINE)


def test_langgraph_is_imported_only_under_the_planner_graphs() -> None:
    allowed = SRC / "studio/planner/graphs"
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if allowed not in path.parents
        and any(m == "langgraph" or m.startswith("langgraph.") for m in _imported_modules(path))
    ]
    assert not offenders, offenders


def test_no_langchain_or_vendor_sdk_is_a_dependency() -> None:
    text = (ROOT / "pyproject.toml").read_text()
    for name in ("langchain", "langchain-openai", "openai", "xai-sdk-chat"):
        assert not re.search(rf'"{name}[>=<~ "]', text), name


@pytest.mark.parametrize(
    "path", sorted(PLANNER.rglob("*.py")), ids=lambda p: str(p.relative_to(SRC))
)
def test_the_planner_imports_ports_only(path: Path) -> None:
    for module in _imported_modules(path):
        root = module.split(".")[0]
        assert root not in {"httpx", "sqlalchemy"}, (path.name, module)
        assert not module.startswith("src.api"), (path.name, module)
        assert not module.startswith("src.db"), (path.name, module)


def test_the_adapter_names_no_vendor() -> None:
    text = (SRC / "studio/llm/openai_compatible.py").read_text().replace("OpenAICompatible", "")
    assert not re.search(r"xai|grok|qwen|dashscope|aliyun|openai", text, flags=re.IGNORECASE)


def test_no_model_id_or_vendor_literal_in_the_studio_runs_code() -> None:
    files = [
        *SRC.glob("studio/llm/*.py"),
        *SRC.glob("studio/planner/**/*.py"),
        SRC / "studio/run_requests.py",
        SRC / "studio/run_failures.py",
        SRC / "studio/services/run.py",
        SRC / "studio/services/run_executor.py",
        SRC / "studio/services/run_context.py",
        SRC / "core/llm_models.py",
        SRC / "workers/studio_run.py",
    ]
    for path in files:
        literals = _code_string_literals(path)
        assert not [
            lit for lit in literals if re.search(r"grok|qwen|xai|dashscope|gpt-|claude-", lit, re.I)
        ], path


def test_the_tracing_switches_are_named_only_by_the_startup_guard() -> None:
    offenders = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if re.search(r"LANGSMITH_TRACING|LANGCHAIN_TRACING_V2", path.read_text())
    ]
    assert offenders == ["core/llm_tracing.py"]


def test_the_graphs_package_pins_tracing_before_langgraph_is_imported() -> None:
    init = (PLANNER / "graphs/__init__.py").read_text()
    assert "force_tracing_off()" in init
    assert "langgraph" not in "".join(
        line for line in init.splitlines() if line.startswith(("import", "from"))
    )


def test_llm_origin_is_written_in_exactly_one_place() -> None:
    assert EXECUTOR.read_text().count("origin=StudioDocumentOrigin.LLM") == 1
    assert "StudioDocumentOrigin" not in RUN_SERVICE.read_text()


def test_broad_excepts_exist_only_as_the_documented_run_guards() -> None:
    """The executor's claim and run guards are the only places a broad error is caught."""
    assert not re.search(r"except\s+(StudioError|Exception)", RUN_SERVICE.read_text())
    executor = EXECUTOR.read_text()
    assert len(re.findall(r"except\s+Exception", executor)) == 2  # claim guard + run guard
    assert len(re.findall(r"except\s+StudioError", executor)) == 1  # the save step's refusals
    assert re.findall(r"except\s+VersionUnchanged", executor) == ["except VersionUnchanged"]
    assert not re.search(r"except\s*:", executor)


def test_the_active_run_index_is_named_in_the_migration_and_matched_in_the_service() -> None:
    migration = (ROOT / "alembic/versions/057_studio_runs.py").read_text()
    assert "uq_studio_runs_one_active_per_track" in migration
    assert "uq_studio_runs_one_active_per_track" in (SRC / "db/models/studio.py").read_text()
    assert "RUN_ONE_ACTIVE_PER_TRACK" in RUN_SERVICE.read_text()


def test_the_migration_is_057_after_056() -> None:
    text = (ROOT / "alembic/versions/057_studio_runs.py").read_text()
    assert 'revision: str = "057"' in text
    assert re.search(r'down_revision: str \| Sequence\[str\] \| None = "056"', text)


def test_the_run_workers_lease_name_is_stable() -> None:
    assert 'WORKER_NAME: Final = "studio_run"' in (SRC / "workers/studio_run.py").read_text()


def test_the_runs_are_only_wired_behind_the_runs_active_switch() -> None:
    common = (SRC / "api/dependencies/common.py").read_text()
    assert common.count("_start_studio_run_worker(") == 2  # the definition and the one call
    block = common[common.index("if settings.studio_runs_active:") :]
    assert "_start_studio_run_worker(" in block.split("# Initialize Web Push")[0]


def test_the_contract_has_a_planner_runs_section() -> None:
    contract = (ROOT / "docs/contracts/studio-contract.md").read_text()
    assert len(re.findall(r"^## 11\. Planner runs", contract, flags=re.MULTILINE)) == 1
    for needle in (
        "studio.run.updated",
        "run_in_progress",
        "run_not_cancellable",
        "idempotency_conflict",
        "plan_invalid",
        "run_timeout",
        "content_policy_refused",
        "Idempotency-Key",
        "never made current",
    ):
        assert needle in contract, needle
