"""N1: no SQL parameter reaches an exception message or an error log, against real PostgreSQL.

The failing statements are real ones - PostgreSQL refuses U+0000 in ``text`` and in ``jsonb`` -
run through the engine the app itself builds (``DatabaseManager``). The log assertions hold even
if the API's own refusal of U+0000 were removed.

The promise covers SQL *parameters*. It does not cover the ``DETAIL`` line PostgreSQL adds to a
constraint violation, which the chained driver exception prints.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest
import pytest_asyncio
import structlog
from litestar.testing import RequestFactory
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.pool import NullPool
from structlog.testing import capture_logs

from src.api.app import global_exception_handler
from src.db.session import DatabaseManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

    from sqlalchemy.ext.asyncio import AsyncEngine

MARKER = "SECRET-LYRIC-MARKER"
VALUE = f"{MARKER}\x00tail"
_HIDDEN_NOTICE = "SQL parameters hidden due to hide_parameters=True"

# The two column types that hold user text, and the parameter each gets.
_COLUMNS = [
    pytest.param("text", ":value", VALUE, id="text"),
    pytest.param(
        "jsonb", "CAST(:value AS jsonb)", f'{{"line": "{MARKER}\\u0000tail"}}', id="jsonb"
    ),
]


@pytest_asyncio.fixture
async def app_engine(test_database_url: str) -> AsyncGenerator[AsyncEngine]:
    """The engine exactly as the app builds it. The probe table is temporary: no migrations."""
    manager = DatabaseManager(test_database_url, pool_size=1, max_overflow=0)
    yield manager.engine
    await manager.close()


async def _failing_insert(
    engine: AsyncEngine, column: str, placeholder: str, value: str
) -> DBAPIError:
    """Insert ``value`` into a throwaway table and return what the driver raised."""
    async with engine.connect() as conn:
        await conn.execute(text(f"CREATE TEMPORARY TABLE nul_probe (body {column}) ON COMMIT DROP"))
        with pytest.raises(DBAPIError) as raised:
            await conn.execute(
                text(f"INSERT INTO nul_probe (body) VALUES ({placeholder})"),  # noqa: S608
                {"value": value},
            )
        await conn.rollback()
    return raised.value


def _rendered_unhandled_exception(exc: Exception) -> str:
    """What ``global_exception_handler`` logs, rendered as the JSON log format renders it."""
    request = RequestFactory().post("/v1/studio/tracks")
    with capture_logs() as logs:
        global_exception_handler(request, exc)

    (event,) = [log for log in logs if log["event"] == "unhandled_exception"]
    assert event["exc_info"] is exc, "the traceback must be what gets logged"
    rendered: dict[str, Any] = dict(structlog.processors.format_exc_info(None, "error", event))
    assert "Traceback" in rendered["exception"]
    return repr(rendered)


@pytest.mark.parametrize(("column", "placeholder", "value"), _COLUMNS)
async def test_the_exception_message_carries_no_parameter(
    app_engine: AsyncEngine, column: str, placeholder: str, value: str
) -> None:
    exc = await _failing_insert(app_engine, column, placeholder, value)

    message = str(exc)
    assert MARKER not in message
    assert "[parameters:" not in message
    assert _HIDDEN_NOTICE in message


@pytest.mark.parametrize(("column", "placeholder", "value"), _COLUMNS)
async def test_the_unhandled_exception_log_carries_no_text(
    app_engine: AsyncEngine, column: str, placeholder: str, value: str
) -> None:
    exc = await _failing_insert(app_engine, column, placeholder, value)

    logged = _rendered_unhandled_exception(exc)

    assert MARKER not in logged
    assert "[parameters:" not in logged


@pytest.mark.parametrize(("column", "placeholder", "value"), _COLUMNS)
async def test_the_log_assertion_can_fail(
    test_database_url: str, column: str, placeholder: str, value: str
) -> None:
    """The control: the same failure through an engine that does not hide parameters leaks the
    marker into the same rendered log, so the test above is not passing for some other reason."""
    engine = create_async_engine(test_database_url, poolclass=NullPool)
    try:
        exc = await _failing_insert(engine, column, placeholder, value)
    finally:
        await engine.dispose()

    assert MARKER in str(exc)
    assert MARKER in _rendered_unhandled_exception(exc)


async def test_the_suite_engine_hides_parameters_like_production(db_engine: AsyncEngine) -> None:
    assert db_engine.sync_engine.hide_parameters is True
