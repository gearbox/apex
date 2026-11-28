"""R1: ``format_exception_safely`` renders every traceback without a database value.

The chains here are built by hand, so no database is needed; the real violations are in
``tests/integration/test_sql_parameters_hidden.py``. Frame lines quote source code, so the values
are referred to by name and never written out on a line that raises.
"""

from __future__ import annotations

import sys
import traceback
from typing import TYPE_CHECKING

import asyncpg
import pytest
from sqlalchemy.dialects.postgresql.asyncpg import AsyncAdapt_asyncpg_dbapi
from sqlalchemy.exc import DBAPIError, IntegrityError, PendingRollbackError, StatementError

from src.core import exception_format
from src.core.exception_format import format_exception_safely

if TYPE_CHECKING:
    from collections.abc import Callable

MARKER = "SECRET-ROW-MARKER"
STATEMENT = "INSERT INTO studio_artist_chat_messages (content) VALUES ($1::VARCHAR)"
CONSTRAINT = "ck_studio_chat_messages_user_complete"
IDENTIFIERS = (
    f"sqlstate=23514 constraint_name={CONSTRAINT} "
    "table_name=studio_artist_chat_messages schema_name=public"
)


def _check_violation() -> asyncpg.exceptions.CheckViolationError:
    """What asyncpg raises, with the row in ``DETAIL`` as PostgreSQL reports it."""
    error = asyncpg.exceptions.CheckViolationError(f"new row violates a check constraint {MARKER}")
    error.detail = f"Failing row contains (7, {MARKER}, null)."
    error.hint = f"hint {MARKER}"
    error.constraint_name = CONSTRAINT
    error.table_name = "studio_artist_chat_messages"
    error.schema_name = "public"
    return error


def _raise_integrity_error() -> None:
    """asyncpg error, raised from by SQLAlchemy's ``IntegrityError``."""
    driver_error = _check_violation()
    integrity_error = IntegrityError(
        STATEMENT, {"content": MARKER}, Exception(f"driver said {MARKER}"), hide_parameters=True
    )
    try:
        raise driver_error
    except asyncpg.exceptions.CheckViolationError as raised:
        raise integrity_error from raised


def _raised(raiser: Callable[[], object]) -> BaseException:
    try:
        raiser()
    except BaseException as exc:  # the exception itself is the fixture
        return exc
    pytest.fail("nothing was raised")


def _safely(exc: BaseException) -> str:
    return format_exception_safely((type(exc), exc, exc.__traceback__))


def _as_the_stdlib(exc: BaseException) -> str:
    return "".join(traceback.format_exception(exc)).removesuffix("\n")


def _assert_nothing_of_the_row(rendered: str) -> None:
    assert MARKER not in rendered
    assert "DETAIL" not in rendered
    assert "Failing row" not in rendered


# ---------------------------------------------------------------------------
# A database chain
# ---------------------------------------------------------------------------


def test_a_database_chain_shows_identifiers_the_statement_and_the_frames() -> None:
    rendered = _safely(_raised(_raise_integrity_error))

    _assert_nothing_of_the_row(rendered)
    # The last exception: class, identifiers gathered from what it wraps, then the statement.
    assert rendered.endswith(f"sqlalchemy.exc.IntegrityError: {IDENTIFIERS}\n[SQL: {STATEMENT}]")
    assert f"asyncpg.exceptions.CheckViolationError: {IDENTIFIERS}\n" in rendered
    assert "The above exception was the direct cause of the following exception" in rendered
    assert rendered.count("Traceback (most recent call last):") == 2
    assert f'File "{__file__}", line ' in rendered
    assert "in _raise_integrity_error" in rendered
    assert "raise integrity_error from raised" in rendered
    # The statement is the only thing kept of the SQLAlchemy text.
    assert "[parameters" not in rendered
    assert "sqlalche.me" not in rendered


def test_the_plain_traceback_of_the_same_chain_holds_the_text() -> None:
    """The control: this chain does leak when ``format_exception_safely`` is not what renders it."""
    rendered = _as_the_stdlib(_raised(_raise_integrity_error))

    assert MARKER in rendered
    assert "DETAIL:  Failing row contains" in rendered


def test_an_asyncpg_error_alone_is_a_database_error() -> None:
    def raiser() -> None:
        raise _check_violation()

    rendered = _safely(_raised(raiser))

    _assert_nothing_of_the_row(rendered)
    assert rendered.endswith(f"asyncpg.exceptions.CheckViolationError: {IDENTIFIERS}")


@pytest.mark.parametrize("name", ["Error", "IntegrityError", "UniqueViolationError"])
def test_sqlalchemys_asyncpg_adapter_errors_are_database_errors(name: str) -> None:
    """They subclass no ``StatementError``: only the module says what they are."""
    adapter_class = getattr(AsyncAdapt_asyncpg_dbapi, name)

    def raiser() -> None:
        raise adapter_class(f"Key (name)=({MARKER}) already exists.")

    rendered = _safely(_raised(raiser))

    assert MARKER not in rendered
    assert rendered.endswith(
        f"sqlalchemy.dialects.postgresql.asyncpg.AsyncAdapt_asyncpg_dbapi.{name}"
    )


def test_what_a_database_error_was_raised_from_is_withheld_too() -> None:
    """asyncpg quotes a parameter it cannot encode, and so does the codec error beneath it."""

    def encode() -> None:
        try:
            raise TypeError(f"codec context {MARKER}")  # noqa: TRY301 - building a chain
        except TypeError:
            # No "from": the codec's own context, still part of the driver's report.
            raise ValueError(f"invalid UUID {MARKER!r}: length must be 32")  # noqa: B904

    def bind() -> None:
        try:
            encode()
        except ValueError as exc:
            message = f"invalid input for query argument $1: {MARKER!r} ({exc})"
            raise asyncpg.exceptions.DataError(message) from exc

    def execute() -> None:
        try:
            bind()
        except asyncpg.exceptions.DataError as exc:
            raise DBAPIError(STATEMENT, {}, exc, hide_parameters=True) from exc

    exc = _raised(execute)

    rendered = _safely(exc)

    assert MARKER in _as_the_stdlib(exc)
    assert MARKER not in rendered
    assert "\nTypeError\n" in rendered
    assert "\nValueError\n" in rendered
    assert "asyncpg.exceptions.DataError: sqlstate=22000\n" in rendered


def test_a_statement_error_with_no_driver_error_is_withheld_with_its_cause() -> None:
    """SQLAlchemy's own wrapper of a bind-processing failure quotes whatever that failure said."""

    def raiser() -> None:
        try:
            raise TypeError(f"cannot serialize {MARKER}")  # noqa: TRY301 - building a chain
        except TypeError as exc:
            raise StatementError(
                f"(builtins.TypeError) cannot serialize {MARKER}", STATEMENT, {}, exc
            ) from exc

    rendered = _safely(_raised(raiser))

    assert MARKER not in rendered
    assert rendered.endswith(f"sqlalchemy.exc.StatementError\n[SQL: {STATEMENT}]")


def test_pending_rollback_error_repeats_a_database_error_and_is_withheld() -> None:
    def raiser() -> None:
        raise PendingRollbackError(
            "This Session's transaction has been rolled back due to a previous exception during "
            f"flush. Original exception was: invalid input syntax for type integer: {MARKER!r}"
        )

    rendered = _safely(_raised(raiser))

    assert MARKER not in rendered
    assert rendered.endswith("sqlalchemy.exc.PendingRollbackError")


def test_a_database_error_inside_an_exception_group_is_withheld() -> None:
    def raiser() -> None:
        database_error = _raised(_raise_integrity_error)
        assert isinstance(database_error, Exception)
        raise ExceptionGroup("two tasks failed", [KeyError("kept-key"), database_error])

    exc = _raised(raiser)

    rendered = _safely(exc)

    assert MARKER in _as_the_stdlib(exc)
    _assert_nothing_of_the_row(rendered)
    assert "ExceptionGroup: two tasks failed (2 sub-exceptions)" in rendered
    assert "KeyError: 'kept-key'" in rendered
    assert f"| sqlalchemy.exc.IntegrityError: {IDENTIFIERS}\n" in rendered


def test_a_database_error_reached_through_context_is_withheld() -> None:
    """A failure while handling a database error: the link is ``__context__``, not ``__cause__``."""

    def raiser() -> None:
        try:
            _raise_integrity_error()
        except IntegrityError:
            raise RuntimeError("the handler failed too")  # noqa: B904

    exc = _raised(raiser)
    assert exc.__cause__ is None
    assert isinstance(exc.__context__, IntegrityError)

    rendered = _safely(exc)

    _assert_nothing_of_the_row(rendered)
    assert "During handling of the above exception, another exception occurred" in rendered
    assert f"sqlalchemy.exc.IntegrityError: {IDENTIFIERS}" in rendered
    assert rendered.endswith("RuntimeError: the handler failed too")


def test_what_the_application_was_handling_keeps_its_message() -> None:
    """A database error's own context is the application's, not the driver's."""

    def raiser() -> None:
        try:
            raise ValueError("provider timed out")  # noqa: TRY301 - building a chain
        except ValueError:
            raise _check_violation()  # noqa: B904

    rendered = _safely(_raised(raiser))

    _assert_nothing_of_the_row(rendered)
    assert "ValueError: provider timed out\n" in rendered


def test_an_exception_reached_both_ways_is_withheld_wherever_it_is_printed() -> None:
    """What is withheld is decided over the whole graph before anything is printed: a codec error
    also listed as a plain group member is not shown there just because that copy comes first."""

    def raiser() -> None:
        shared = ValueError(f"invalid UUID {MARKER!r}")
        database_error = DBAPIError(STATEMENT, {}, shared, hide_parameters=True)
        database_error.__cause__ = shared
        raise ExceptionGroup("both ways", [shared, database_error])

    rendered = _safely(_raised(raiser))

    assert MARKER not in rendered
    assert "| ValueError\n" in rendered


def test_notes_on_a_database_error_are_not_rendered() -> None:
    def raiser() -> None:
        error = _check_violation()
        error.add_note(f"while saving {MARKER}")
        raise error

    _assert_nothing_of_the_row(_safely(_raised(raiser)))


# ---------------------------------------------------------------------------
# Everything else is the stdlib's output
# ---------------------------------------------------------------------------


def _plain() -> None:
    raise ValueError("the message survives")


def _with_cause() -> None:
    try:
        _plain()
    except ValueError as exc:
        raise KeyError("outer") from exc


def _with_context() -> None:
    try:
        _plain()
    except ValueError:
        raise KeyError("outer")  # noqa: B904


def _with_suppressed_context() -> None:
    try:
        _plain()
    except ValueError:
        raise KeyError("outer") from None


def _with_notes() -> None:
    error = ValueError("noted")
    error.add_note("first note")
    error.add_note("second note")
    raise error


def _never_raised_cause() -> None:
    raise RuntimeError("outer") from ValueError("never raised, so it has no frames")


def _syntax_error() -> None:
    compile("def broken(:\n    pass", "<broken>", "exec")


def _group() -> None:
    members = [_raised(_with_cause), _raised(_with_context), OSError("bare")]
    raise ExceptionGroup("outer group", [m for m in members if isinstance(m, Exception)])


def _nested_groups() -> None:
    first, second = _raised(_group), _raised(_group)
    assert isinstance(first, Exception)
    assert isinstance(second, Exception)
    try:
        raise ExceptionGroup("middle", [first, ValueError("sibling")])  # noqa: TRY301
    except ExceptionGroup as middle:
        raise ExceptionGroup("top", [second, OSError("last")]) from middle


def _group_as_cause() -> None:
    try:
        _group()
    except ExceptionGroup as exc:
        raise RuntimeError("after the group") from exc


def _wide_group() -> None:
    raise ExceptionGroup("wide", [ValueError(str(index)) for index in range(18)])


def _deep_group() -> None:
    error: Exception = ValueError("bottom")
    for level in range(13):
        error = ExceptionGroup(f"level {level}", [error])
    raise error


def _context_cycle() -> None:
    first, second = ValueError("first"), ValueError("second")
    first.__context__, second.__context__ = second, first
    raise first


@pytest.mark.parametrize(
    "raiser",
    [
        _plain,
        _with_cause,
        _with_context,
        _with_suppressed_context,
        _with_notes,
        _never_raised_cause,
        _syntax_error,
        _group,
        _nested_groups,
        _group_as_cause,
        _wide_group,
        _deep_group,
        _context_cycle,
    ],
)
def test_without_a_database_error_the_output_is_the_stdlibs(
    raiser: Callable[[], None],
) -> None:
    """Ordinary debugging is unchanged: every message, frame, link and group, character for
    character."""
    exc = _raised(raiser)

    assert _safely(exc) == _as_the_stdlib(exc)


def test_the_traceback_given_is_the_one_rendered_for_the_last_exception() -> None:
    """structlog hands over a triple; its traceback wins over the exception's own."""
    try:
        _plain()
    except ValueError:
        triple = sys.exc_info()
    exc_type, exc, tb = triple
    assert exc_type is not None
    assert exc is not None
    exc.__traceback__ = None

    assert "in _plain" in format_exception_safely((exc_type, exc, tb))


# ---------------------------------------------------------------------------
# It never raises
# ---------------------------------------------------------------------------


def test_a_broken_formatter_withholds_all_text_instead_of_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If it raised, ``logging`` would print the record itself, exception ``repr`` and all."""

    def broken(_: BaseException) -> frozenset[int]:
        raise ZeroDivisionError

    exc = _raised(_raise_integrity_error)
    monkeypatch.setattr(exception_format, "_withheld_ids", broken)

    rendered = _safely(exc)

    _assert_nothing_of_the_row(rendered)
    assert "in _raise_integrity_error" in rendered
    assert "sqlalchemy.exc.IntegrityError\n" in rendered
    assert "traceback formatting failed (ZeroDivisionError); exception text withheld" in rendered
