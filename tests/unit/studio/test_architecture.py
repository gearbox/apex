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
        SRC / "db/models/studio_cast.py",
        SRC / "db/repositories/studio.py",
        SRC / "db/repositories/studio_cast.py",
        SRC / "db/repositories/studio_track_audio.py",
        SRC / "db/repositories/studio_timeline.py",
        *SRC.glob("api/routes/studio_*.py"),
        SRC / "api/schemas/studio.py",
        SRC / "api/schemas/studio_runs.py",
        SRC / "api/schemas/studio_chat.py",
        SRC / "api/security/studio.py",
        SRC / "db/models/studio_chat.py",
        SRC / "db/repositories/studio_chat.py",
        SRC / "workers/studio_run.py",
        SRC / "workers/studio_chat.py",
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


MEDIA_INGEST_FILES = sorted(SRC.glob("api/services/media_ingest/*.py"))


def test_the_file_sets_are_not_empty() -> None:
    assert len(LIBRARY_FILES) >= 8
    assert len(STUDIO_FILES) >= 20
    assert len(MEDIA_INGEST_FILES) >= 8


@pytest.mark.parametrize("path", LIBRARY_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_the_library_never_imports_the_studio(path: Path) -> None:
    offending = {
        m for m in _imported_modules(path) if m == "src.studio" or m.startswith("src.studio.")
    }
    assert not offending, f"{path.relative_to(ROOT)} imports {offending}"


@pytest.mark.parametrize("path", MEDIA_INGEST_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_media_ingest_never_imports_the_studio(path: Path) -> None:
    """Ingest belongs to the library's side (D12): it knows no bounded context above it."""
    offending = {
        m for m in _imported_modules(path) if m == "src.studio" or m.startswith("src.studio.")
    }
    assert not offending, f"{path.relative_to(ROOT)} imports {offending}"


_STUDIO_FORBIDDEN_IMPORTS = (
    "src.api.services.media_ingest",
    "src.api.services.storage",
    "src.api.services.user_content",
)


@pytest.mark.parametrize("path", STUDIO_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_the_studio_never_calls_ingest_storage_or_the_upload_service(path: Path) -> None:
    """The studio reaches assets only through ``AssetHoldService`` and the library's builders:
    a file is uploaded first (``POST /v1/storage/upload``) and attached by its ref."""
    offending = {
        m
        for m in _imported_modules(path)
        if any(m == name or m.startswith(f"{name}.") for name in _STUDIO_FORBIDDEN_IMPORTS)
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


@pytest.mark.parametrize(
    "path",
    [SRC / "db/repositories/studio_timeline.py", SRC / "studio/services/timeline.py"],
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_the_timeline_code_never_flushes_or_commits(path: Path) -> None:
    """Timeline writes belong to the request transaction (S3a)."""
    assert not re.search(r"session\.(commit|flush|rollback)\(", path.read_text()), path


@pytest.mark.parametrize(
    "path",
    [
        SRC / "db/repositories/studio_cast.py",
        SRC / "db/repositories/library_hold.py",
        SRC / "api/services/library_holds.py",
        SRC / "studio/services/cast.py",
    ],
    ids=lambda p: str(p.relative_to(SRC)),
)
def test_the_cast_and_hold_code_never_flushes_or_commits(path: Path) -> None:
    """The request transaction is the caller's: holds and cast rows commit (or not) together."""
    assert not re.search(r"session\.(commit|flush|rollback)\(", path.read_text()), path


# -- S1-P3: the library owns the holds, any context may use them (D12) ---------------------------


def test_the_new_library_modules_are_under_the_import_rule() -> None:
    """A rename must not silently drop them from the ``library never imports studio`` check."""
    names = {path.relative_to(SRC).as_posix() for path in LIBRARY_FILES}
    assert {
        "api/services/library_holds.py",
        "api/services/library_media.py",
        "db/repositories/library_hold.py",
        "db/models/library.py",
    } <= names


def test_the_library_hold_modules_name_no_holder_by_type() -> None:
    """A holder is an opaque ``(kind, id)``: no studio model, repository or error is imported."""
    for name in ("api/services/library_holds.py", "db/repositories/library_hold.py"):
        imported = _imported_modules(SRC / name)
        assert not {m for m in imported if "studio" in m}, name


def test_only_the_hold_service_writes_the_holds_table() -> None:
    """The cast reads ``library_asset_holds`` (a join) and writes it only through the service."""
    allowed = {"db/repositories/library_hold.py"}
    writers = re.compile(r"(pg_insert|insert|delete|update)\(\s*LibraryAssetHold\b")
    for path in SRC.rglob("*.py"):
        name = path.relative_to(SRC).as_posix()
        if name in allowed:
            continue
        assert not writers.search(path.read_text()), name


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
    assert len(controllers) == 8
    for path in controllers:
        text = path.read_text()
        if path.name == "studio_runs.py":  # the run switch sits between the feature guard and auth
            assert "guards = [studio_feature_guard, studio_runs_guard, auth_guard]" in text, path
        elif path.name == "studio_chat.py":  # likewise the chat switch
            assert "guards = [studio_feature_guard, studio_chat_guard, auth_guard]" in text, path
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
    """The planner imports ``src.llm.port`` only - never the adapter, the registry or the factory."""
    for module in _imported_modules(path):
        root = module.split(".")[0]
        assert root not in {"httpx", "sqlalchemy"}, (path.name, module)
        assert not module.startswith("src.api"), (path.name, module)
        assert not module.startswith("src.db"), (path.name, module)
        if module == "src.llm" or module.startswith("src.llm."):
            assert module == "src.llm.port", (path.name, module)


def test_no_studio_module_but_the_executor_sees_the_registry_or_the_factory() -> None:
    private = {"src.llm.registry", "src.llm.factory", "src.llm.openai_compatible"}
    importers = sorted(
        str(path.relative_to(SRC))
        for path in SRC.glob("studio/**/*.py")
        if private & _imported_modules(path)
    )
    assert importers == ["studio/services/run_executor.py"]


def test_the_adapter_names_no_vendor() -> None:
    text = (SRC / "llm/openai_compatible.py").read_text().replace("OpenAICompatible", "")
    assert not re.search(r"xai|grok|qwen|dashscope|aliyun|openai", text, flags=re.IGNORECASE)


def test_no_model_id_or_vendor_literal_in_the_studio_runs_code() -> None:
    files = [
        *SRC.glob("llm/*.py"),
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


RUN_KINDS = SRC / "studio/services/run_kinds"


def test_llm_origin_is_written_in_exactly_one_place() -> None:
    """Every kind's result goes through ``create_llm_version`` (``run_kinds/base.py``)."""
    assert "StudioDocumentOrigin" not in EXECUTOR.read_text()
    assert "StudioDocumentOrigin" not in RUN_SERVICE.read_text()
    writers = {
        str(path.relative_to(SRC)): path.read_text().count("origin=StudioDocumentOrigin.LLM")
        for path in SRC.glob("studio/**/*.py")
        if "origin=StudioDocumentOrigin.LLM" in path.read_text()
    }
    assert writers == {"studio/services/run_kinds/base.py": 1}


def test_broad_excepts_exist_only_as_the_documented_run_guards() -> None:
    """The executor's claim and run guards are the only places a broad error is caught."""
    assert not re.search(r"except\s+(StudioError|Exception)", RUN_SERVICE.read_text())
    executor = EXECUTOR.read_text()
    assert len(re.findall(r"except\s+Exception", executor)) == 2  # claim guard + run guard
    assert len(re.findall(r"except\s+StudioError", executor)) == 1  # the save step's refusals
    assert not re.search(r"except\s*:", executor)
    # The two things a save may legitimately swallow live in the shared save helpers, once each.
    base = (RUN_KINDS / "base.py").read_text()
    assert re.findall(r"except\s+VersionUnchanged", base) == ["except VersionUnchanged"]
    assert re.findall(r"except\s+ParentVersionNotFound", base) == ["except ParentVersionNotFound"]
    for path in RUN_KINDS.glob("*.py"):
        assert not re.search(r"except\s+(StudioError|Exception|BaseException)", path.read_text())


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


def test_the_shared_llm_access_is_built_in_one_place_and_only_when_required() -> None:
    """The registry and the HTTP client exist only behind ``llm_required`` (invariant 22)."""
    common = (SRC / "api/dependencies/common.py").read_text()
    assert common.count("LlmModelRegistry.from_settings(") == 1
    tree = ast.parse(common)
    (builder,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_init_llm_access"
    ]
    guard = builder.body[1]  # after the docstring
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == "not settings.llm_required"
    assert isinstance(guard.body[0], ast.Return)
    assert ast.unparse(builder).count("httpx.AsyncClient(") == 1  # the one shared client
    # the worker starter is handed the client; it never makes its own
    (starter,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "_start_studio_run_worker"
    ]
    assert "AsyncClient(" not in ast.unparse(starter)
    # the tracing guard stays with the planner switch
    runs_block = common[common.index("if settings.studio_runs_active:") :]
    assert "force_tracing_off()" in runs_block.split("# Initialize Web Push")[0]


def test_the_llm_requirement_is_one_setting() -> None:
    config = (SRC / "core/config.py").read_text()
    assert len(re.findall(r"def llm_required\(self\)", config)) == 1


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


# -- S1-P2a-A remediation R2: no assertions in guarded code (invariant 23) -----------------------

NO_ASSERT_GLOBS = (
    "llm/**/*.py",
    "studio/planner/**/*.py",
    "studio/services/run*.py",
    "studio/services/memory*.py",
    "studio/services/run_kinds/*.py",
    "studio/run_*.py",
    "workers/studio_run.py",
    # the artist chat (S1-P2c): its task bodies run under broad guards too
    "studio/chat/**/*.py",
    "studio/services/chat*.py",
    "workers/studio_chat.py",
)


def _no_assert_files() -> list[Path]:
    return sorted({path for pattern in NO_ASSERT_GLOBS for path in SRC.glob(pattern)})


def _assertions(tree: ast.AST) -> list[int]:
    """Line numbers of ``assert`` statements and ``raise AssertionError`` (bare or called)."""
    lines: list[int] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            lines.append(node.lineno)
        elif isinstance(node, ast.Raise) and node.exc is not None:
            target = node.exc.func if isinstance(node.exc, ast.Call) else node.exc
            if isinstance(target, ast.Name) and target.id == "AssertionError":
                lines.append(node.lineno)
    return lines


def test_the_no_assert_file_set_covers_the_guarded_modules() -> None:
    names = {str(path.relative_to(SRC)) for path in _no_assert_files()}
    assert {
        "llm/openai_compatible.py",
        "studio/planner/graphs/plan.py",
        "studio/services/run_executor.py",
        "studio/services/run.py",
        "studio/services/run_kinds/base.py",
        "studio/services/run_kinds/lyrics.py",
        "studio/planner/graphs/lyrics.py",
        "studio/run_failures.py",
        "studio/run_requests.py",
        "workers/studio_run.py",
        "studio/chat/supervisor.py",
        "studio/chat/sweep.py",
        "studio/services/chat.py",
        "studio/services/memory_draft.py",
        "workers/studio_chat.py",
    } <= names


@pytest.mark.parametrize("path", _no_assert_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_no_assert_and_no_raise_assertion_error_in_guarded_code(path: Path) -> None:
    """A broken invariant must not be reported as an ordinary run failure, nor vanish under -O."""
    assert _assertions(ast.parse(path.read_text())) == [], path


class TestTheAssertionDetector:
    """The rule is read from the syntax tree: comments and strings cannot trip it."""

    @pytest.mark.parametrize(
        "source",
        [
            "assert x is not None",
            "assert x, 'message'",
            "raise AssertionError('unreachable')",
            "raise AssertionError",
            "def f():\n    if y:\n        assert z\n",
        ],
    )
    def test_it_finds(self, source: str) -> None:
        assert _assertions(ast.parse(source))

    @pytest.mark.parametrize(
        "source",
        [
            "# assert x\nx = 1",
            "text = 'assert x is not None'",
            "text = 'raise AssertionError'",
            'def f():\n    """assert in a docstring"""\n',
            "raise RuntimeError('x')",
            "raise PlannerStateError('validate ran before draft')",
        ],
    )
    def test_it_ignores(self, source: str) -> None:
        assert _assertions(ast.parse(source)) == []


def test_the_rule_is_recorded_in_claude_md() -> None:
    text = (ROOT / "CLAUDE.md").read_text()
    assert "no `assert`" in text
    assert "PlannerStateError" in text


def test_the_executor_catches_a_timeout_only_around_the_graph() -> None:
    """A timeout from the save step is not a ``run_timeout`` (R2.4)."""
    tree = ast.parse(EXECUTOR.read_text())
    handlers = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.ExceptHandler)
        and isinstance(node.type, ast.Name)
        and node.type.id == "TimeoutError"
    ]
    assert len(handlers) == 1
    (handler,) = handlers
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    guarded = parents[handler]
    assert isinstance(guarded, ast.Try)
    assert "_run_graph" in ast.unparse(guarded.body[0])
    assert "_finish_success" not in "".join(ast.unparse(stmt) for stmt in guarded.body)


# -- S1-P2a-B: the handler seam, lyrics runs ------------------------------------------------------


def test_the_executor_names_no_run_kind() -> None:
    """P2a-48: one executor, one handler per kind; nothing kind-specific leaks back in."""
    text = EXECUTOR.read_text()
    for needle in (
        "PlanRunRequest",
        "LyricsRunRequest",
        "ReviseLyricsRunRequest",
        "compile_plan_graph",
        "StudioRunKind.PLAN",
        "StudioRunKind.LYRICS",
        "StudioRunKind.REVISE_LYRICS",
        "PlanReady",
        "LyricsReady",
        "prompts.",
    ):
        assert needle not in text, needle
    assert not [m for m in _imported_modules(EXECUTOR) if "run_kinds.registry" in m]


def test_every_run_kind_has_a_handler() -> None:
    from src.core.enums import StudioRunKind
    from src.studio.services.run_kinds.registry import default_handlers

    handlers = default_handlers()
    assert set(handlers) == set(StudioRunKind)
    assert all(handler.kind is kind for kind, handler in handlers.items())


def test_the_request_side_never_loads_a_graph() -> None:
    """The API process imports ``run.py`` and ``intake.py``: no handler, no graph, no langgraph."""
    for path in (RUN_SERVICE, RUN_KINDS / "intake.py"):
        for module in _imported_modules(path):
            assert "planner.graphs" not in module, (path.name, module)
            assert module.split(".")[0] != "langgraph", (path.name, module)
            for handler in ("run_kinds.plan", "run_kinds.lyrics", "run_kinds.revise_lyrics"):
                assert not module.endswith(handler), (path.name, module)
            assert "run_kinds.registry" not in module, (path.name, module)


def test_only_the_wiring_imports_the_handler_registry() -> None:
    importers = sorted(
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if any(m.endswith("run_kinds.registry") for m in _imported_modules(path))
    )
    assert importers == ["api/dependencies/common.py"]


def test_the_package_init_of_the_handlers_imports_nothing() -> None:
    assert _imported_modules(RUN_KINDS / "__init__.py") == set()


def test_run_kind_not_available_is_gone() -> None:
    for path in [*SRC.rglob("*.py"), ROOT / "docs/contracts/studio-contract.md"]:
        assert not re.search(r"RunKindNotAvailable|run_kind_not_available", path.read_text()), path


def test_the_migration_is_058_after_057() -> None:
    text = (ROOT / "alembic/versions/058_studio_run_inputs.py").read_text()
    assert 'revision: str = "058"' in text
    assert re.search(r'down_revision: str \| Sequence\[str\] \| None = "057"', text)


def test_the_snapshot_is_version_3() -> None:
    assert re.search(r"snapshot_version: Literal\[3\]", (PLANNER / "context.py").read_text())


def test_the_syllable_counters_have_one_home() -> None:
    text = (PLANNER / "syllables.py").read_text()
    assert len(re.findall(r"^def counter_for\b", text, flags=re.MULTILINE)) == 1
    others = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name != "syllables.py" and re.search(r"def counter_for\b", path.read_text())
    ]
    assert others == []


def test_the_planner_modules_of_lyrics_are_pure() -> None:
    """Counters, checks, mapping and review import neither the database nor the graph library."""
    for name in ("syllables", "lyric_checks", "lyrics_mapping", "lyrics_review"):
        for module in _imported_modules(PLANNER / f"{name}.py"):
            assert module.split(".")[0] not in {"langgraph", "sqlalchemy", "httpx"}, (name, module)
            assert not module.startswith(("src.api", "src.db")), (name, module)


def test_the_version_in_use_guard_is_in_the_errors_the_service_and_the_contract() -> None:
    assert "version_in_use_by_run" in (SRC / "studio/errors.py").read_text()
    assert "VersionInUseByRun" in (SRC / "studio/services/version.py").read_text()
    assert "version_in_use_by_run" in (ROOT / "docs/contracts/studio-contract.md").read_text()


# -- S1-P2b: artist memories ----------------------------------------------------------------------


def test_the_memories_module_is_pure_and_the_selection_has_one_home() -> None:
    """Rendering and selecting import no database, API or LLM-adapter module (invariant: pure)."""
    for module in _imported_modules(PLANNER / "memories.py"):
        assert module.split(".")[0] not in {"langgraph", "sqlalchemy", "httpx"}, module
        assert not module.startswith(("src.api", "src.db", "src.llm")), module
    text = (PLANNER / "memories.py").read_text()
    for name in ("select_memories", "render_memory", "rendered_size"):
        assert len(re.findall(rf"^def {name}\b", text, flags=re.MULTILINE)) == 1, name
    others = [
        str(path.relative_to(SRC))
        for path in SRC.rglob("*.py")
        if path.name != "memories.py"
        and re.search(r"def (select_memories|render_memory)\b", path.read_text())
    ]
    assert others == []


def test_nothing_under_the_renderers_knows_a_memory() -> None:
    """Memories are planner-only (P2-M6): no renderer imports or names a memory type."""
    for path in (SRC / "studio/renderers").rglob("*.py"):
        text = path.read_text()
        assert not re.search(r"(?i)memor", text), path
        assert not {m for m in _imported_modules(path) if m.startswith("src.studio.planner")}, path


def test_the_memory_limits_are_defined_once_in_constraints() -> None:
    constants = (
        "MEMORY_TITLE_MAX",
        "MEMORY_BODY_MAX",
        "MEMORY_THEMES_MAX",
        "MEMORY_THEME_MAX",
        "MEMORY_IMAGERY_MAX",
        "MEMORY_IMAGE_MAX",
        "MAX_ACTIVE_MEMORIES_PER_ARTIST",
        "MEMORY_CONTEXT_BUDGET",
    )
    for name in constants:
        definers = [
            str(path.relative_to(SRC))
            for path in SRC.rglob("*.py")
            if re.search(rf"^{name}\b\s*[:=]", path.read_text(), flags=re.MULTILINE)
        ]
        assert definers == ["studio/constraints.py"], (name, definers)
    assert "MEMORY_CONTEXT_BUDGET" in (SRC / "studio/services/run_kinds/intake.py").read_text()
    assert "MEMORY_CONTEXT_BUDGET" in (PLANNER / "memories.py").read_text()


def test_memories_not_available_is_gone() -> None:
    for path in [*SRC.rglob("*.py"), ROOT / "docs/contracts/studio-contract.md"]:
        assert not re.search(r"MemoriesNotAvailable|memories_not_available", path.read_text()), path


def test_the_migration_is_059_after_058() -> None:
    text = (ROOT / "alembic/versions/059_studio_artist_memories.py").read_text()
    assert 'revision: str = "059"' in text
    assert re.search(r'down_revision: str \| Sequence\[str\] \| None = "058"', text)
    assert not re.search(r"^\s*(from|import)\s+src\b", text, flags=re.MULTILINE)


def test_the_prompt_versions_are_v2() -> None:
    expected = {
        "plan_v2.py": "plan-v2",
        "lyrics_v2.py": "lyrics-v2",
        "revise_lyrics_v2.py": "revise-lyrics-v2",
    }
    for name, version in expected.items():
        text = (PLANNER / "prompts" / name).read_text()
        assert f'PROMPT_VERSION: Final = "{version}"' in text, name
    for path in (SRC / "studio/services/run_kinds").glob("*.py"):
        assert not re.search(r"prompts\.\w+_v1\b", path.read_text()), path
    for path in (PLANNER / "graphs").glob("*.py"):
        assert not re.search(r"prompts\.\w+_v1\b", path.read_text()), path


def test_the_memory_lock_order_is_documented_where_the_locks_are_taken() -> None:
    service = (SRC / "studio/services/memory.py").read_text()
    assert "artist, then memory" in service
    # the PATCH locks the artist before the memory, in that order in the source as well
    patch = service[service.index("async def patch") :]
    assert patch.index("self._artists.lock") < patch.index("self._memories.lock")
    # the memory repository never locks an artist (that would invert the order)
    repo = (SRC / "db/repositories/studio.py").read_text()
    section = repo[
        repo.index("class StudioArtistMemoryRepository") : repo.index("class ProjectListRow")
    ]
    assert "StudioArtist." not in section  # no artist join: the memory row carries its scope
    assert "_owned_by" not in section
    assert "with_for_update(of=StudioArtistMemory)" in section
