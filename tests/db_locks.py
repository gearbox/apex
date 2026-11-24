"""Proving, in PostgreSQL itself, that a specific request is waiting on a lock.

Concurrency tests start two overlapping requests and need to know the second one is really blocked
behind the first before they let the first one finish. Asking "is *any* session waiting?" is not
that proof: a session that has not reached its lock yet (a task that starts late, a connection
still being opened) is invisible to it, and so is the difference between the session under test and
an unrelated one. So the test names the session by the **backend pid** it reported
(:func:`backend_pid`) and asks the database about exactly that backend.

Every probe runs in ``AUTOCOMMIT``: ``pg_stat_activity`` is cached for the length of a transaction,
so a probe that polls inside one transaction keeps reading the first snapshot and can never see a
backend that starts waiting afterwards.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

__all__ = ["backend_pid", "wait_until_blocked", "wait_until_blocked_by"]

_POLL_SECONDS = 0.01

_STATE_OF = text(
    "SELECT cardinality(pg_blocking_pids(pid)) AS blockers, state, wait_event_type "
    "FROM pg_stat_activity WHERE pid = :pid"
)
_BLOCKED_BY = text(
    "SELECT pid FROM pg_stat_activity WHERE :blocker = ANY(pg_blocking_pids(pid)) ORDER BY pid"
)


async def backend_pid(session: AsyncSession) -> int:
    """The PostgreSQL backend pid that serves ``session``'s connection.

    A racing request reports this first and then does its work **in the same session**, so the
    test can later watch exactly that backend.
    """
    pid: int = (await session.execute(text("SELECT pg_backend_pid()"))).scalar_one()
    return pid


async def wait_until_blocked(
    engine: AsyncEngine,
    pid: int,
    *,
    timeout: float = 10.0,  # noqa: ASYNC109 - a deadline that reports what the backend was doing
) -> None:
    """Return once backend ``pid`` waits on a lock (``pg_blocking_pids(pid)`` is not empty).

    Fails the test when the backend is gone (its request ended without ever waiting) or when it
    has not blocked within ``timeout`` seconds; the failure names the backend's state, never a
    query.

    Args:
        engine: Any engine on the database under test (the probe uses its own connection).
        pid: The backend to watch, as reported by :func:`backend_pid`.
        timeout: How long to wait for the backend to block, in seconds.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    last = "never seen"
    async with engine.execution_options(isolation_level="AUTOCOMMIT").connect() as probe:
        while loop.time() < deadline:
            row = (await probe.execute(_STATE_OF, {"pid": pid})).one_or_none()
            if row is None:
                pytest.fail(
                    f"backend {pid} is gone: its request ended without ever waiting for a lock"
                )
            if row.blockers:
                return
            last = f"state={row.state}, wait_event_type={row.wait_event_type}"
            await asyncio.sleep(_POLL_SECONDS)
    pytest.fail(f"backend {pid} did not wait for a lock within {timeout:g} s ({last})")


async def wait_until_blocked_by(
    engine: AsyncEngine,
    blocker_pid: int,
    *,
    timeout: float = 10.0,  # noqa: ASYNC109 - a deadline that reports what it was waiting for
) -> int:
    """Return the pid of a backend that waits on a lock **held by** backend ``blocker_pid``.

    For a test whose second request is started by a helper that does not report its own pid: what
    is known is the first request, the one that holds the lock.

    Args:
        engine: Any engine on the database under test.
        blocker_pid: The backend that holds the lock, as reported by :func:`backend_pid`.
        timeout: How long to wait, in seconds.

    Returns:
        The pid of the (lowest-numbered) backend blocked by ``blocker_pid``.
    """
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    async with engine.execution_options(isolation_level="AUTOCOMMIT").connect() as probe:
        while loop.time() < deadline:
            blocked = (await probe.execute(_BLOCKED_BY, {"blocker": blocker_pid})).scalars().all()
            if blocked:
                return int(blocked[0])
            await asyncio.sleep(_POLL_SECONDS)
    pytest.fail(f"no backend waited on a lock held by backend {blocker_pid} within {timeout:g} s")
