"""The chat's structural rules, as durable tests (read from the syntax tree, never a text grep)."""

from __future__ import annotations

import ast
import re
from pathlib import Path
from uuid import UUID

import msgspec
import pytest
from structlog.testing import capture_logs

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SRC = ROOT / "src"
CHAT = SRC / "studio" / "chat"
CHAT_FILES = sorted(CHAT.rglob("*.py"))
CHAT_LOGGING_FILES = sorted(
    {
        *CHAT_FILES,
        SRC / "studio/services/chat.py",
        SRC / "studio/services/chat_context.py",
        SRC / "studio/services/chat_events.py",
        SRC / "studio/services/memory_draft.py",
        SRC / "studio/services/memory_selection.py",
        SRC / "workers/studio_chat.py",
        SRC / "llm/openai_compatible.py",
    }
)


def _runtime_imports(path: Path) -> set[str]:
    """Modules imported at run time: whatever sits under ``if TYPE_CHECKING:`` is not."""
    tree = ast.parse(path.read_text())
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and ast.unparse(node.test) in {
            "TYPE_CHECKING",
            "typing.TYPE_CHECKING",
        }:
            guarded.update(id(child) for child in ast.walk(node))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if id(node) in guarded:
            continue
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def test_the_chat_package_is_not_empty() -> None:
    names = {path.name for path in CHAT_FILES}
    assert {"supervisor.py", "stream.py", "window.py", "discography.py", "sweep.py"} <= names


@pytest.mark.parametrize("path", CHAT_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_the_chat_package_imports_no_api_module_at_run_time(path: Path) -> None:
    offending = {
        m
        for m in _runtime_imports(path)
        if m == "src.api" or m.startswith(("src.api.", "src.workers"))
    }
    assert not offending, f"{path.relative_to(ROOT)} imports {offending}"


@pytest.mark.parametrize("path", CHAT_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_the_chat_package_never_sees_the_registry_the_factory_or_the_adapter(path: Path) -> None:
    private = {"src.llm.registry", "src.llm.factory", "src.llm.openai_compatible"}
    assert not private & _runtime_imports(path)


def test_the_planner_never_imports_the_chat() -> None:
    """P2c-6: the planner never reads chat."""
    for path in (SRC / "studio" / "planner").rglob("*.py"):
        offending = {m for m in _runtime_imports(path) if m.startswith("src.studio.chat")}
        assert not offending, path


def test_the_pure_chat_modules_import_no_database() -> None:
    pure = ["stream.py", "window.py", "discography.py", "draft.py", "failures.py", "jobs.py"]
    for name in pure:
        modules = _runtime_imports(CHAT / name)
        assert not {m for m in modules if m.startswith(("src.db", "sqlalchemy", "httpx"))}, name


# -- privacy: nothing here logs message, draft or memory text (invariant 15) --------------------

_LOGGER_METHODS = frozenset({"debug", "info", "warning", "warn", "error", "exception", "critical"})
_ALLOWED_FIELDS = frozenset(
    {
        # ids
        "chat_id",
        "message_id",
        "user_message_id",
        "artist_message_id",
        "source_message_id",
        "draft_id",
        "artist_id",
        "user_id",
        "memory_id",
        # names of things, codes, flags and counts - never text
        "task",
        "status",
        "error_code",
        "model_key",
        "origin",
        "exc_type",
        "frames",
        "abandoned",
        "count",
        "content_length",
        "memory_count",
        "price_tokens",
        "prompt_tokens",
        "completion_tokens",
        "truncated",
        "duration_ms",
        "status_code",
        "finish_reason",
        "delivered",
        "attempt",
    }
)


def _is_logger_call(node: ast.Call) -> bool:
    func = node.func
    return (
        isinstance(func, ast.Attribute)
        and func.attr in _LOGGER_METHODS
        and isinstance(func.value, ast.Name)
        and func.value.id in {"logger", "log", "_logger"}
    )


def _logging_violations(tree: ast.AST) -> list[str]:
    found: list[str] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_logger_call(node)):
            continue
        event = node.args[0] if node.args else None
        if not (isinstance(event, ast.Constant) and isinstance(event.value, str)):
            found.append(f"line {node.lineno}: a log event must be a constant string")
        if len(node.args) > 1:
            found.append(f"line {node.lineno}: log fields must be keywords")
        found.extend(
            f"line {node.lineno}: log field {kw.arg!r} is not an allowed field"
            for kw in node.keywords
            if kw.arg not in _ALLOWED_FIELDS
        )
    return found


@pytest.mark.parametrize("path", CHAT_LOGGING_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_chat_code_logs_ids_counts_and_codes_only(path: Path) -> None:
    assert _logging_violations(ast.parse(path.read_text())) == [], path


class TestTheLoggingCheckSeesWhatItShould:
    @pytest.mark.parametrize(
        "source",
        [
            'logger.info("x", content=message.content)',
            'logger.info("x", text=text)',
            'logger.info("x", body=body)',
            'logger.info("x", title=chat.title)',
            "logger.info(event)",
            'logger.info("x", chat_id=1, **fields)',
        ],
    )
    def test_it_finds(self, source: str) -> None:
        assert _logging_violations(ast.parse(source))

    @pytest.mark.parametrize(
        "source",
        [
            'logger.info("x", message_id=str(i), status=s, error_code=c, duration_ms=1)',
            'logger.warning("x", task=t, abandoned=2)',
        ],
    )
    def test_it_ignores(self, source: str) -> None:
        assert _logging_violations(ast.parse(source)) == []


async def test_the_event_bus_logs_no_payload(monkeypatch: pytest.MonkeyPatch) -> None:
    """A delta carries the artist's words: ``EventBus.publish`` must never put a payload in a log."""
    from uuid import UUID

    from src.api.schemas.events import EventType, StudioChatDeltaPayload
    from src.api.services import event_bus as module

    published: list[bytes] = []

    class _Redis:
        async def publish(self, _channel: str, data: bytes) -> None:
            published.append(data)

    monkeypatch.setattr(module, "get_redis_client", _Redis)
    secret = "THE-ARTISTS-PRIVATE-WORDS"
    with capture_logs() as logs:
        await module.EventBus(enabled=True).publish(
            user_id=UUID(int=1),
            event_type=EventType.STUDIO_CHAT_DELTA,
            payload=StudioChatDeltaPayload(
                chat_id=UUID(int=2), message_id=UUID(int=3), seq=1, text=secret
            ),
        )
    assert published, "the event was sent"
    assert secret.encode() in published[0], "the text travels to the owner"
    assert secret not in str(logs)


def test_a_delta_payload_is_ids_a_sequence_and_text_only() -> None:
    from src.api.schemas.events import (
        StudioChatDeltaPayload,
        StudioChatMessageUpdatedPayload,
        StudioMemoryDraftUpdatedPayload,
    )

    assert StudioChatDeltaPayload.__struct_fields__ == ("chat_id", "message_id", "seq", "text")
    # the status events never carry text: ids, a status and a revision
    assert StudioChatMessageUpdatedPayload.__struct_fields__ == (
        "chat_id",
        "message_id",
        "status",
        "revision",
    )
    assert StudioMemoryDraftUpdatedPayload.__struct_fields__ == (
        "draft_id",
        "chat_id",
        "source_message_id",
        "status",
        "revision",
    )
    encoded = msgspec.json.encode(
        StudioChatMessageUpdatedPayload(
            chat_id=UUID(int=1), message_id=UUID(int=2), status="complete", revision=2
        )
    )
    assert b"text" not in encoded


# -- wiring and docs ---------------------------------------------------------------------------


def test_the_event_types_are_the_documented_three() -> None:
    from src.api.schemas.events import EventType

    assert EventType.STUDIO_CHAT_DELTA.value == "studio.chat.delta"
    assert EventType.STUDIO_CHAT_MESSAGE_UPDATED.value == "studio.chat.message.updated"
    assert EventType.STUDIO_MEMORY_DRAFT_UPDATED.value == "studio.memory_draft.updated"


def test_the_chat_is_only_wired_behind_the_chat_active_switch() -> None:
    common = (SRC / "api/dependencies/common.py").read_text()
    block = common[common.index("if settings.studio_chat_active:") :]
    block = block.split("# Initialize Web Push")[0]
    assert "_build_studio_chat_supervisor(" in block
    assert "_start_studio_chat_sweep_worker(" in block
    assert common.count("_build_studio_chat_supervisor(") == 2  # the definition and the one call
    assert common.count("_start_studio_chat_sweep_worker(") == 2


def test_the_sweep_workers_lease_name_is_stable() -> None:
    assert (
        'WORKER_NAME: Final = "studio_chat_sweep"' in (SRC / "workers/studio_chat.py").read_text()
    )


def test_migration_060_only_reads_the_runs_for_the_backfill_and_never_imports_application_code() -> (
    None
):
    """A memory draft is its own table, not a kind of planner run: 060 never alters
    ``studio_runs``. The one thing it does with it is the usage ledger's backfill, a single
    ``INSERT … SELECT`` of the runs of the last hour (R2)."""
    source = (ROOT / "alembic/versions/060_studio_artist_chat.py").read_text()
    tree = ast.parse(source)
    docstring = ast.get_docstring(tree, clean=False)
    mentions = [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node.value != docstring
        and "studio_runs" in node.value
    ]
    assert len(mentions) == 1  # the backfill, and nothing else in the code
    (backfill,) = mentions
    assert backfill.startswith("INSERT INTO studio_llm_usage")
    assert "FROM studio_runs WHERE created_at > now() - interval '1 hour'" in backfill
    assert not {"UPDATE", "DELETE", "ALTER", "DROP"} & set(backfill.upper().split())
    assert not re.search(r"^\s*(from|import)\s+src\b", source, flags=re.MULTILINE)
    assert 'revision: str = "060"' in source
    assert 'down_revision: str | Sequence[str] | None = "059"' in source


def test_the_one_pending_reply_index_is_named_in_the_model_the_migration_and_the_service() -> None:
    name = "uq_studio_chat_one_pending_reply"
    assert name in (SRC / "db/models/studio_chat.py").read_text()
    assert name in (ROOT / "alembic/versions/060_studio_artist_chat.py").read_text()
    # the service matches the violated constraint by the model's constant, never by a prior SELECT
    assert "CHAT_ONE_PENDING_REPLY" in (SRC / "studio/services/chat.py").read_text()


def test_the_contract_has_an_artist_chat_section() -> None:
    contract = (ROOT / "docs/contracts/studio-contract.md").read_text()
    assert len(re.findall(r"^## 13\. Artist chat", contract, flags=re.MULTILINE)) == 1
