"""N1 / R1: no row value reaches an error log, against real PostgreSQL.

The failing statements are real ones, run through the engine the app itself builds
(``DatabaseManager``). Two things keep their values out of the log, and each is tested for itself:

* **N1** - ``hide_parameters`` keeps SQL parameters out of the SQLAlchemy message, so even a plain
  traceback of these failures (PostgreSQL refuses U+0000 in ``text`` and in ``jsonb``) has none.
  This holds even if the API's own refusal of U+0000 were removed.
* **R1** - that does not reach what PostgreSQL itself reports: ``DETAIL: Failing row contains
  (...)``, ``Key (...)=(...) already exists``, or the parameter asyncpg quotes when it cannot
  encode it. ``configure_logging`` therefore renders every traceback with
  ``format_exception_safely``: a database error shows as its class and identifiers, never its text.
  These cases go through the real JSON logging setup and read what it writes to stdout.
"""

from __future__ import annotations

import json
import logging
import traceback
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import pytest
import pytest_asyncio
import structlog
from litestar.testing import RequestFactory
from sqlalchemy import text, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool
from structlog.testing import capture_logs

from src.api.app import global_exception_handler
from src.core.config import Settings
from src.core.logging import configure_logging
from src.db.models.storage import GenerationJob
from src.db.models.studio import StudioArtist
from src.db.models.studio_chat import StudioArtistChat, StudioArtistChatMessage
from src.db.models.user import User
from src.db.session import DatabaseManager

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Generator

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


def _plainly_rendered_unhandled_exception(exc: Exception) -> str:
    """What ``global_exception_handler`` logs, with its traceback rendered by structlog's plain
    ``format_exc_info``: no redaction, so only ``hide_parameters`` stands between the parameter
    and the text."""
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

    logged = _plainly_rendered_unhandled_exception(exc)

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
    assert MARKER in _plainly_rendered_unhandled_exception(exc)


async def test_the_suite_engine_hides_parameters_like_production(db_engine: AsyncEngine) -> None:
    assert db_engine.sync_engine.hide_parameters is True


# ---------------------------------------------------------------------------
# R1: what PostgreSQL and asyncpg say about a row never reaches the log
# ---------------------------------------------------------------------------

# Short and early in its column: PostgreSQL cuts each value in DETAIL at 64 bytes.
ROW_TEXT = "secret-row-marker"


@pytest_asyncio.fixture
async def app_session(
    db_engine: AsyncEngine,  # noqa: ARG001 - it runs the migrations
    app_engine: AsyncEngine,
) -> AsyncGenerator[AsyncSession]:
    """A session on the app's own engine, over the migrated schema. Nothing is committed."""
    async with AsyncSession(app_engine, expire_on_commit=False) as session:
        yield session
        await session.rollback()


async def _new_user(session: AsyncSession) -> User:
    user = User(
        id=uuid4(),
        email=f"redaction-{uuid4().hex}@example.com",
        password_hash="x",
        product_id="vex",
    )
    session.add(user)
    await session.flush()
    return user


def _artist(user: User, name: str) -> StudioArtist:
    return StudioArtist(id=uuid4(), user_id=user.id, product_id="vex", name=name)


async def _violate_not_null(session: AsyncSession) -> None:
    """A failed job UPDATE: the row PostgreSQL then prints holds the prompt."""
    user = await _new_user(session)
    job = GenerationJob(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        provider="grok",
        generation_type="t2i",
        status="running",
        name="Probe",
        prompt=ROW_TEXT,
    )
    session.add(job)
    await session.flush()
    await session.execute(update(GenerationJob).where(GenerationJob.id == job.id).values(name=None))


async def _violate_check(session: AsyncSession) -> None:
    """A studio backstop CHECK on a chat message, whose row holds the message text."""
    user = await _new_user(session)
    artist = _artist(user, "Probe")
    session.add(artist)
    await session.flush()
    chat = StudioArtistChat(
        id=uuid4(), user_id=user.id, product_id="vex", artist_id=artist.id, title="Probe"
    )
    session.add(chat)
    await session.flush()
    session.add(
        StudioArtistChatMessage(
            id=uuid4(),
            user_id=user.id,
            product_id="vex",
            chat_id=chat.id,
            position=1,
            role="user",
            status="pending",  # a user message is born complete
            content=ROW_TEXT,
            client_message_id=uuid4(),
        )
    )
    await session.flush()


async def _violate_unique(session: AsyncSession) -> None:
    """Two artists of one name: ``Key (..., lower(name))=(..., <the name>) already exists``."""
    user = await _new_user(session)
    session.add(_artist(user, ROW_TEXT))
    await session.flush()
    session.add(_artist(user, ROW_TEXT))
    await session.flush()


async def _bind_what_asyncpg_cannot_encode(session: AsyncSession) -> None:
    """Refused before it is sent: asyncpg quotes the parameter, ``hide_parameters`` or not."""
    await session.execute(text("SELECT CAST(:value AS uuid)"), {"value": ROW_TEXT})


_VIOLATIONS = [
    pytest.param(
        _violate_not_null,
        "sqlstate=23502 table_name=generation_jobs column_name=name schema_name=public",
        id="not-null",
    ),
    pytest.param(
        _violate_check,
        "sqlstate=23514 constraint_name=ck_studio_chat_messages_user_complete "
        "table_name=studio_artist_chat_messages schema_name=public",
        id="check-on-a-chat-message",
    ),
    pytest.param(
        _violate_unique,
        "sqlstate=23505 constraint_name=uq_studio_artists_owner_name "
        "table_name=studio_artists schema_name=public",
        id="unique",
    ),
    pytest.param(_bind_what_asyncpg_cannot_encode, "sqlstate=22000", id="unencodable-parameter"),
]


async def _failure(
    session: AsyncSession, violate: Callable[[AsyncSession], Awaitable[None]]
) -> DBAPIError:
    with pytest.raises(DBAPIError) as raised:
        await violate(session)
    await session.rollback()
    return raised.value


@pytest.fixture
def _restore_logging() -> Generator[None]:
    """Undo ``configure_logging``: structlog's setup, and the root handler it binds to stdout.

    The processors list is restored in place: loggers cached by earlier tests hold that object.
    """
    old = structlog.get_config()
    processors = old["processors"]
    snapshot = list(processors)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    root.handlers, root.level = handlers, level
    processors[:] = snapshot
    structlog.configure(**{**old, "processors": processors})


@pytest.mark.usefixtures("_restore_logging")
@pytest.mark.parametrize(("violate", "identifiers"), _VIOLATIONS)
async def test_the_json_log_of_a_database_error_holds_no_row_value(
    app_session: AsyncSession,
    violate: Callable[[AsyncSession], Awaitable[None]],
    identifiers: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    exc = await _failure(app_session, violate)
    # Before configure_logging: the factory builds a default Litestar app, which sets up stdlib
    # logging of its own and would take the root handler back.
    request = RequestFactory().post("/v1/studio/tracks")
    configure_logging(Settings(comfyui_host="127.0.0.1", log_level="INFO", log_format="json"))
    capsys.readouterr()

    global_exception_handler(request, exc)

    output = capsys.readouterr().out
    (event,) = [
        line
        for line in map(json.loads, output.splitlines())
        if line["event"] == "unhandled_exception"
    ]
    assert ROW_TEXT not in output.lower()
    assert "DETAIL" not in output
    assert "[parameters:" not in output
    # What is left is enough to find the failure: identifiers, the statement and the frames.
    assert identifiers in event["exception"]
    assert "[SQL: " in event["exception"]
    assert "Traceback (most recent call last):" in event["exception"]
    assert f'File "{__file__}", line ' in event["exception"]


@pytest.mark.parametrize(("violate", "identifiers"), _VIOLATIONS)
async def test_the_plain_traceback_of_a_database_error_holds_the_row_value(
    app_session: AsyncSession,
    violate: Callable[[AsyncSession], Awaitable[None]],
    identifiers: str,  # noqa: ARG001 - the same cases as the test above
) -> None:
    """The control: through ``traceback.format_exception`` every one of these leaks, on the same
    engine with its parameters hidden, so the test above passes because of the formatter."""
    exc = await _failure(app_session, violate)

    assert ROW_TEXT in "".join(traceback.format_exception(exc)).lower()
