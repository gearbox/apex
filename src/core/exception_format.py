"""Render a traceback without any database value.

``src.core.logging.configure_logging`` renders every logged traceback through
:func:`format_exception_safely`.

PostgreSQL puts row values into an error: ``DETAIL:  Failing row contains (...)`` on a NOT NULL or
CHECK violation, ``Key (...)=(...) already exists`` on a unique one, and sometimes the primary
message itself (``invalid input syntax for type integer: "..."``). asyncpg quotes a parameter it
cannot encode, which ``hide_parameters`` does not touch. The driver exception that holds all this
is chained into every traceback. So the rule goes by exception type, never by a pattern over the
text, which would have to know every message PostgreSQL can produce: **a database exception's text
is never rendered; its identifiers are.**

The layout is ``traceback.format_exception``'s own: for a chain with no database error in it the
output is the same, character for character.
"""

from __future__ import annotations

import textwrap
import traceback
from typing import TYPE_CHECKING, Final

from sqlalchemy.exc import PendingRollbackError, StatementError

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from types import TracebackType

    from structlog.typing import ExcInfo

__all__ = ["format_exception_safely"]

# Names of schema objects and the error code: all that is shown of a database error.
_IDENTIFIERS: Final = (
    "sqlstate",
    "constraint_name",
    "table_name",
    "column_name",
    "schema_name",
    "data_type_name",
)

_TRACEBACK_HEADER: Final = "Traceback (most recent call last):\n"
_CAUSE_LINK: Final = "\nThe above exception was the direct cause of the following exception:\n\n"
_CONTEXT_LINK: Final = "\nDuring handling of the above exception, another exception occurred:\n\n"
# The stdlib's own limits for exception groups.
_MAX_GROUP_WIDTH: Final = 15
_MAX_GROUP_DEPTH: Final = 10


def format_exception_safely(exc_info: ExcInfo) -> str:
    """Render ``exc_info`` like ``traceback.format_exception``, withholding database text.

    Every frame, the ``__cause__`` / ``__context__`` chain and ``ExceptionGroup`` members are
    rendered. Frame locals never are. In place of its message a database error shows its class,
    whichever of its identifiers exist (``sqlstate``, ``constraint_name``, ...) and, for a
    ``StatementError``, the statement, which holds placeholders and not values. The message of
    whatever a database error was raised from is withheld too: it is the driver's own account of
    the failure and quotes the same values.

    This never raises. A formatter that raises makes ``logging`` print the record itself,
    exception ``repr`` included.

    Args:
        exc_info: The ``(type, value, traceback)`` triple structlog hands to its formatter.

    Returns:
        The rendered traceback, without a trailing newline.
    """
    _, exc, tb = exc_info
    try:
        rendered = "".join(_Renderer(_withheld_ids(exc)).format(exc, tb))
    except Exception as failure:  # fail closed: frames and class names only
        rendered = "".join(
            [
                _TRACEBACK_HEADER,
                *traceback.format_tb(tb),
                f"{_qualified_name(type(exc))}\n",
                (
                    f"[traceback formatting failed ({type(failure).__name__}); "
                    "exception text withheld]"
                ),
            ]
        )
    return rendered.removesuffix("\n")


def _is_database_error(exc: BaseException) -> bool:
    """Whether ``exc`` carries what the database or its driver said."""
    # PendingRollbackError is neither kind of database class, but its message repeats the text of
    # the failure that deactivated the session.
    if isinstance(exc, StatementError | PendingRollbackError):
        return True
    # asyncpg's own exceptions, and SQLAlchemy's adapter errors in ``...postgresql.asyncpg``.
    return any("asyncpg" in cls.__module__.split(".") for cls in type(exc).__mro__)


def _withheld_ids(root: BaseException) -> frozenset[int]:
    """``id()`` of every exception under ``root`` whose message must not be rendered.

    Those are the database errors, everything they were raised from, and the members of a withheld
    group. A database error's own ``__context__`` is not part of its report: it is whatever the
    application was handling when the statement failed, and it keeps its message. Below an
    exception that is already withheld without being a database error (a codec's ``ValueError``)
    the context is the driver's as well.
    """
    withheld: set[int] = set()
    # An exception can be reached both ways, so visits are told apart by what they inherit.
    visited: set[tuple[int, bool]] = set()
    pending: list[tuple[BaseException, bool]] = [(root, False)]
    while pending:
        exc, inherited = pending.pop()
        if (id(exc), inherited) in visited:
            continue
        visited.add((id(exc), inherited))
        is_database = _is_database_error(exc)
        hidden = inherited or is_database
        if hidden:
            withheld.add(id(exc))
        if exc.__cause__ is not None:
            pending.append((exc.__cause__, hidden))
        if exc.__context__ is not None:
            pending.append((exc.__context__, hidden and not is_database))
        if isinstance(exc, BaseExceptionGroup):
            pending.extend((member, hidden) for member in exc.exceptions)
    return frozenset(withheld)


def _qualified_name(cls: type[BaseException]) -> str:
    """The class name as a traceback prints it."""
    if cls.__module__ in ("__main__", "builtins"):
        return cls.__qualname__
    return f"{cls.__module__}.{cls.__qualname__}"


def _identifiers(exc: BaseException) -> list[str]:
    """``name=value`` for each identifier on ``exc`` or on what it wraps (``orig`` / cause).

    SQLAlchemy's adapter error keeps only the ``sqlstate``; the constraint, table and column are
    on the asyncpg error beneath it. Only database errors are read.
    """
    found: dict[str, str] = {}
    seen: set[int] = set()
    wrapped: list[BaseException] = [exc]
    for current in wrapped:  # grows while it is walked
        if id(current) in seen:
            continue
        seen.add(id(current))
        if _is_database_error(current):
            for name in _IDENTIFIERS:
                value = getattr(current, name, None)
                if isinstance(value, str) and value:
                    found.setdefault(name, value)
        wrapped.extend(
            inner
            for inner in (getattr(current, "orig", None), current.__cause__)
            if isinstance(inner, BaseException)
        )
    return [f"{name}={found[name]}" for name in _IDENTIFIERS if name in found]


def _redacted_message(exc: BaseException) -> str:
    """What stands where a withheld exception's message would be."""
    line = _qualified_name(type(exc))
    if _is_database_error(exc) and (identifiers := _identifiers(exc)):
        line = f"{line}: {' '.join(identifiers)}"
    if isinstance(exc, StatementError) and exc.statement:
        line = f"{line}\n[SQL: {exc.statement}]"
    return f"{line}\n"


class _Renderer:
    """``traceback.TracebackException.format``, printing a message only where it may be.

    Written out rather than patched into ``TracebackException`` so that it fails closed: a
    withheld exception is never turned into text at all.
    """

    def __init__(self, withheld: frozenset[int]) -> None:
        self._withheld = withheld
        self._seen: set[int] = set()
        self._depth = 0
        self._need_close = False

    def format(self, exc: BaseException, tb: TracebackType | None) -> Iterator[str]:
        """``exc`` and the chain behind it, oldest first."""
        chain: list[tuple[str | None, BaseException, TracebackType | None]] = []
        while True:
            self._seen.add(id(exc))
            link, earlier = self._earlier(exc)
            chain.append((link, exc, tb))
            if earlier is None:
                break
            exc, tb = earlier, earlier.__traceback__
        for link, item, item_tb in reversed(chain):
            if link is not None:
                yield from self._emit([link])
            if isinstance(item, BaseExceptionGroup):
                yield from self._format_group(item, item_tb)
            else:
                yield from self._format_one(item, item_tb)

    def _earlier(self, exc: BaseException) -> tuple[str | None, BaseException | None]:
        cause, context = exc.__cause__, exc.__context__
        if cause is not None and id(cause) not in self._seen:
            return _CAUSE_LINK, cause
        if context is not None and not exc.__suppress_context__ and id(context) not in self._seen:
            return _CONTEXT_LINK, context
        return None, None

    def _message(self, exc: BaseException) -> list[str]:
        if id(exc) in self._withheld:
            return [_redacted_message(exc)]
        return traceback.format_exception_only(exc)

    def _emit(self, chunks: Iterable[str], margin: str = "|") -> Iterator[str]:
        indent = " " * (2 * self._depth)
        if self._depth:
            indent += f"{margin} "
        for chunk in chunks:
            yield textwrap.indent(chunk, indent, lambda _: True)

    def _format_one(self, exc: BaseException, tb: TracebackType | None) -> Iterator[str]:
        if frames := traceback.format_tb(tb):
            yield from self._emit([_TRACEBACK_HEADER])
            yield from self._emit(frames)
        yield from self._emit(self._message(exc))

    def _format_group(
        self, group: BaseExceptionGroup[BaseException], tb: TracebackType | None
    ) -> Iterator[str]:
        if self._depth > _MAX_GROUP_DEPTH:
            yield from self._emit([f"... (max_group_depth is {_MAX_GROUP_DEPTH})\n"])
            return
        toplevel = self._depth == 0
        if toplevel:
            self._depth = 1
        if frames := traceback.format_tb(tb):
            yield from self._emit(
                [f"Exception Group {_TRACEBACK_HEADER}"], margin="+" if toplevel else "|"
            )
            yield from self._emit(frames)
        yield from self._emit(self._message(group))

        total = len(group.exceptions)
        shown = min(total, _MAX_GROUP_WIDTH + 1)
        self._need_close = False
        for index in range(shown):
            last = index == shown - 1
            if last:
                # A nested group may draw the closing line first.
                self._need_close = True
            truncated = index >= _MAX_GROUP_WIDTH
            title = "..." if truncated else str(index + 1)
            lead = "+-" if index == 0 else "  "
            yield f"{' ' * (2 * self._depth)}{lead}+---------------- {title} ----------------\n"
            self._depth += 1
            if truncated:
                more = total - _MAX_GROUP_WIDTH
                yield from self._emit([f"and {more} more exception{'s' if more > 1 else ''}\n"])
            else:
                member = group.exceptions[index]
                yield from self.format(member, member.__traceback__)
            if last and self._need_close:
                yield f"{' ' * (2 * self._depth)}+------------------------------------\n"
                self._need_close = False
            self._depth -= 1

        if toplevel:
            self._depth = 0
