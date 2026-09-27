"""Helpers for translating database integrity violations at service boundaries."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlalchemy.exc import IntegrityError


def violated_constraint(exc: IntegrityError, candidates: frozenset[str]) -> str | None:
    """Return the candidate constraint or unique-index name behind ``exc``, if known.

    asyncpg places the driver exception under ``__cause__``.  Other adapters
    expose only their rendered message, so that fallback deliberately matches
    quoted candidate names only: an unquoted substring could conflate names
    with a shared prefix.
    """
    cause: BaseException | None = exc.orig
    messages: list[str] = [str(exc)]
    seen: set[int] = set()
    while cause is not None and id(cause) not in seen:
        seen.add(id(cause))
        constraint_name = getattr(cause, "constraint_name", None)
        if isinstance(constraint_name, str) and constraint_name in candidates:
            return constraint_name
        messages.append(str(cause))
        cause = cause.__cause__

    for candidate in candidates:
        quoted = f'"{candidate}"'
        if any(quoted in message for message in messages):
            return candidate
    return None
