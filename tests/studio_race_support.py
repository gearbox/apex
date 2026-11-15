"""Deterministic interleaving of a request with a second session (no sleeps, no threads).

A request that reads a track and then the versions its pointers name is exposed to a concurrent
"switch current + delete the old version" between the two reads. ``after_first_track_read`` makes
that interleaving exact: it runs an action once, right after the first statement on a session that
reads ``studio_tracks`` has returned (its rows are already buffered, so the action cannot change
them, but anything the request reads *afterwards* sees the action's committed changes).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sqlalchemy.ext.asyncio import AsyncSession

__all__ = ["after_first_track_read"]


def after_first_track_read(session: AsyncSession, action: Callable[[], Awaitable[object]]) -> None:
    """Await ``action()`` once, right after the first ``studio_tracks`` read on ``session``."""
    original = session.execute
    fired = False

    async def execute(statement: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal fired
        result = await original(statement, *args, **kwargs)
        if not fired and "FROM studio_tracks" in str(statement):
            fired = True
            await action()
        return result

    session.execute = execute  # type: ignore[method-assign]
