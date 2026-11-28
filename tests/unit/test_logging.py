"""Tests for src/core/logging.py — configure_logging and get_logger."""

from __future__ import annotations

import ast
import importlib
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal
from unittest.mock import patch

import asyncpg
import pytest
import structlog
from sqlalchemy.exc import IntegrityError
from structlog.contextvars import bind_contextvars, clear_contextvars, get_contextvars
from structlog.testing import capture_logs

from src.core.config import Settings
from src.core.logging import _THIRD_PARTY_LOGGER_LEVELS, configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import Callable, Generator

_DEFAULT_COMFYUI_PORT: int = Settings.model_fields["comfyui_port"].default


@pytest.fixture
def _reset_structlog() -> Generator[None]:
    """Reset structlog configuration to pre-test state after each test.

    Restores the processors list IN-PLACE (same object, original contents) rather
    than calling reset_defaults(), which creates a NEW list and silently breaks any
    BoundLogger that was cached while cache_logger_on_first_use=True: the cached
    logger holds a reference to the OLD list, so capture_logs() — which modifies
    the NEW list — never reaches it.

    Root cause: app = create_app() runs at collection time, calling configure_logging()
    (cache_logger_on_first_use=True). In make test-all, integration tests exercise
    the service logger first, caching it against list_A. reset_defaults() creates
    list_C; capture_logs() patches list_C; the cached logger still uses list_A.
    In-place restoration keeps list_A as the live list throughout the session.
    """
    old_config = structlog.get_config()
    old_procs_list = old_config["processors"]  # reference to the actual list object
    old_procs_snapshot = list(old_procs_list)  # copy of its current contents
    # configure_logging also replaces the root handlers, with one bound to this test's stdout.
    root = logging.getLogger()
    old_handlers, old_level = list(root.handlers), root.level
    yield
    root.handlers, root.level = old_handlers, old_level
    # Restore the list contents in-place so cached loggers (which hold a reference
    # to this same list object) continue to work correctly with capture_logs().
    old_procs_list.clear()
    old_procs_list.extend(old_procs_snapshot)
    structlog.configure(
        processors=old_procs_list,
        wrapper_class=old_config["wrapper_class"],
        context_class=old_config["context_class"],
        logger_factory=old_config["logger_factory"],
        cache_logger_on_first_use=old_config["cache_logger_on_first_use"],
    )
    clear_contextvars()


@pytest.fixture
def minimal_settings() -> Settings:
    """Minimal Settings with explicit log config for logging tests."""
    return Settings(
        comfyui_host="127.0.0.1",
        comfyui_port=_DEFAULT_COMFYUI_PORT,
        log_level="DEBUG",
        log_format="console",
    )


# ---------------------------------------------------------------------------
# configure_logging
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_reset_structlog")
class TestConfigureLogging:
    def test_json_format_runs_without_error(self) -> None:
        settings = Settings(
            comfyui_host="127.0.0.1",
            comfyui_port=_DEFAULT_COMFYUI_PORT,
            log_level="INFO",
            log_format="json",
        )
        configure_logging(settings)  # must not raise

    def test_console_format_runs_without_error(self, minimal_settings: Settings) -> None:
        configure_logging(minimal_settings)  # must not raise

    def test_root_logger_level_is_applied(self) -> None:
        settings = Settings(
            comfyui_host="127.0.0.1",
            comfyui_port=_DEFAULT_COMFYUI_PORT,
            log_level="WARNING",
            log_format="console",
        )
        configure_logging(settings)
        assert logging.getLogger().level == logging.WARNING

    def test_stdlib_bridge_does_not_raise(self, minimal_settings: Settings) -> None:
        """A stdlib logger record should flow through structlog's handler without error."""
        configure_logging(minimal_settings)
        std_logger = logging.getLogger("test.stdlib.bridge")
        std_logger.info("stdlib test message after configure_logging")

    def test_structlog_positional_args_format_event(
        self, minimal_settings: Settings, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(minimal_settings)

        structlog.get_logger("test.structlog.format").info(
            "%s.event",
            "formatted",
            key="value",
        )

        output = capsys.readouterr().out
        assert "formatted.event" in output
        assert "positional_args" not in output


# ---------------------------------------------------------------------------
# R1: every traceback is rendered without database values
# ---------------------------------------------------------------------------

_ROW_TEXT = "SECRET-LYRIC-ROW"


def _database_failure() -> None:
    driver_error = asyncpg.exceptions.NotNullViolationError("null value in column")
    # Into __dict__, as asyncpg's PostgresError.new does: a type checker sees no such attributes.
    vars(driver_error).update(
        detail=f"Failing row contains (9, {_ROW_TEXT}, null).", column_name="name"
    )
    try:
        raise driver_error
    except asyncpg.exceptions.NotNullViolationError as raised:
        raise IntegrityError(
            "INSERT INTO probe (body) VALUES ($1)", {}, raised, hide_parameters=True
        ) from raised


def _log_structlog_exc_info(name: str) -> None:
    try:
        _database_failure()
    except IntegrityError as exc:
        structlog.get_logger(name).error("unhandled_exception", exc_info=exc)


def _log_structlog_exception(name: str) -> None:
    try:
        _database_failure()
    except IntegrityError:
        structlog.get_logger(name).exception("worker.tick_failed")


def _log_foreign_record(name: str) -> None:
    """A stdlib record, as a library or asyncio emits it."""
    try:
        _database_failure()
    except IntegrityError:
        logging.getLogger(name).exception("library failed")


_LOG_SITES = [
    pytest.param(_log_structlog_exc_info, id="structlog-exc_info"),
    pytest.param(_log_structlog_exception, id="structlog-exception"),
    pytest.param(_log_foreign_record, id="stdlib"),
]


def _settings(log_format: Literal["json", "console"]) -> Settings:
    return Settings(
        comfyui_host="127.0.0.1",
        comfyui_port=_DEFAULT_COMFYUI_PORT,
        log_level="INFO",
        log_format=log_format,
    )


@pytest.mark.usefixtures("_reset_structlog")
class TestTracebacksCarryNoDatabaseValues:
    @pytest.mark.parametrize("log", _LOG_SITES)
    def test_json_renders_the_safe_traceback(
        self, log: Callable[[str], None], capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(_settings("json"))

        log(f"test.redaction.json.{log.__name__}")

        (line,) = capsys.readouterr().out.splitlines()
        rendered = json.loads(line)["exception"]
        assert "in _database_failure" in rendered
        assert "asyncpg.exceptions.NotNullViolationError: sqlstate=23502 column_name=name" in (
            rendered
        )
        assert _ROW_TEXT not in line
        assert "DETAIL" not in line

    @pytest.mark.parametrize("log", _LOG_SITES)
    def test_console_renders_the_safe_traceback(
        self, log: Callable[[str], None], capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(_settings("console"))

        log(f"test.redaction.console.{log.__name__}")

        output = capsys.readouterr().out
        assert "in _database_failure" in output
        assert "sqlstate=23502 column_name=name" in output
        assert _ROW_TEXT not in output
        assert "DETAIL" not in output


# ---------------------------------------------------------------------------
# R3: no frame locals in console tracebacks
# ---------------------------------------------------------------------------

_LOCAL_VALUE = "SECRET-LOCAL-LYRIC"


def _raise_holding(value: str) -> None:
    """The value is a local of the raising frame, and on none of the lines a traceback quotes."""
    lyrics = value
    raise RuntimeError(f"render failed after {len(lyrics)} characters")


@pytest.mark.usefixtures("_reset_structlog")
class TestConsoleTracebacksShowNoLocals:
    """``LOG_FORMAT=console`` is the dev setup, and dev logs get pasted into issues and chats."""

    @pytest.mark.parametrize("source", ["structlog", "stdlib"])
    def test_a_local_of_the_raising_frame_is_not_printed(
        self, source: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        configure_logging(_settings("console"))
        name = f"test.locals.{source}"
        logger = structlog.get_logger(name) if source == "structlog" else logging.getLogger(name)

        try:
            _raise_holding(_LOCAL_VALUE)
        except RuntimeError:
            logger.exception("render.failed")

        output = capsys.readouterr().out
        assert _LOCAL_VALUE not in output
        assert "in _raise_holding" in output
        assert "RuntimeError: render failed after 18 characters" in output

    def test_structlogs_default_console_renderer_does_print_it(self) -> None:
        """The control: the same failure through ``ConsoleRenderer()``'s own traceback."""
        with pytest.raises(RuntimeError) as raised:
            _raise_holding(_LOCAL_VALUE)

        rendered = structlog.dev.ConsoleRenderer(colors=False)(
            None, "error", {"event": "render.failed", "exc_info": raised.value}
        )

        assert _LOCAL_VALUE in rendered


class TestNothingElseRendersTracebacks:
    """``configure_logging`` is only the one renderer while nothing sets up another."""

    def test_litestar_installs_no_logging_of_its_own(self) -> None:
        """Its default ``LoggingConfig`` gives the ``litestar`` logger a plain stdlib handler,
        which in debug mode prints every traceback with its database values."""
        app_module = importlib.import_module("src.api.app")

        assert app_module.create_app().logging_config is None

    def test_granians_own_logging_is_off(self) -> None:
        """Granian prints an exception that escapes the app through its own handler."""
        main_module = importlib.import_module("src.main")

        with (
            patch.object(main_module, "get_settings"),
            patch.object(main_module, "configure_logging"),
            patch.object(main_module.granian, "Granian") as server,
        ):
            main_module.main()

        assert server.call_args.kwargs["log_enabled"] is False

    def test_every_command_that_logs_a_traceback_configures_logging(self) -> None:
        """Unconfigured structlog prints a traceback with every frame's locals."""
        cli_dir = Path(__file__).resolve().parents[2] / "src" / "cli"
        logging_tracebacks: list[str] = []
        unconfigured: list[str] = []
        for path in sorted(cli_dir.glob("*.py")):
            calls = [n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.Call)]
            if not any(
                (isinstance(call.func, ast.Attribute) and call.func.attr == "exception")
                or any(keyword.arg == "exc_info" for keyword in call.keywords)
                for call in calls
            ):
                continue
            logging_tracebacks.append(path.name)
            if not any(
                isinstance(call.func, ast.Name) and call.func.id == "configure_logging"
                for call in calls
            ):
                unconfigured.append(path.name)

        assert "refund_empty_completions.py" in logging_tracebacks, "the scan finds nothing"
        assert unconfigured == []


# ---------------------------------------------------------------------------
# Third-party logger levels
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("_reset_structlog")
class TestThirdPartyLoggerLevels:
    @pytest.mark.parametrize("log_level", ["DEBUG", "INFO", "WARNING"])
    @pytest.mark.parametrize("logger_name", sorted(_THIRD_PARTY_LOGGER_LEVELS))
    def test_pinned_regardless_of_log_level(self, logger_name: str, log_level: str) -> None:
        settings = Settings(
            comfyui_host="127.0.0.1",
            comfyui_port=_DEFAULT_COMFYUI_PORT,
            log_level=log_level,
            log_format="json",
        )
        configure_logging(settings)
        assert logging.getLogger(logger_name).level == _THIRD_PARTY_LOGGER_LEVELS[logger_name]


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_object_with_info_method(self) -> None:
        logger = get_logger("test.module")
        assert callable(logger.info)

    def test_returns_object_with_error_method(self) -> None:
        logger = get_logger("test.module")
        assert callable(logger.error)

    def test_returns_object_with_exception_method(self) -> None:
        logger = get_logger("test.module")
        assert callable(logger.exception)

    def test_named_logger_is_usable_with_capture_logs(self) -> None:
        with capture_logs() as cap:
            log = get_logger("test.named")
            log.info("named.event", key="value")
        assert len(cap) == 1
        assert cap[0]["event"] == "named.event"
        assert cap[0]["key"] == "value"


# ---------------------------------------------------------------------------
# Context vars
# ---------------------------------------------------------------------------


class TestContextVars:
    def test_bound_context_is_accessible_via_get_contextvars(self) -> None:
        bind_contextvars(request_id="abc-123")
        ctx = get_contextvars()
        assert ctx["request_id"] == "abc-123"

    def test_bound_context_event_is_captured(self) -> None:
        with capture_logs() as cap:
            bind_contextvars(request_id="abc-123")
            structlog.get_logger().info("ctx.event")
        assert cap[0]["event"] == "ctx.event"

    def test_clear_contextvars_removes_bound_values(self) -> None:
        bind_contextvars(request_id="to-clear")
        clear_contextvars()
        ctx = get_contextvars()
        assert "request_id" not in ctx

    def test_multiple_context_keys_are_all_bound(self) -> None:
        bind_contextvars(request_id="r1", method="GET", path="/test")
        ctx = get_contextvars()
        assert ctx["request_id"] == "r1"
        assert ctx["method"] == "GET"
        assert ctx["path"] == "/test"
