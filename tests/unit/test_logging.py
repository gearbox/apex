"""Tests for src/core/logging.py — configure_logging and get_logger."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pytest
import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars, get_contextvars
from structlog.testing import capture_logs

from src.core.config import Settings
from src.core.logging import _THIRD_PARTY_LOGGER_LEVELS, configure_logging, get_logger

if TYPE_CHECKING:
    from collections.abc import Generator

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
    yield
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
