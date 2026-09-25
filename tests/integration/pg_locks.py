"""Deterministic synchronisation on PostgreSQL lock waits for concurrency tests.

``asyncio`` events can only say a coroutine is *about to* send a lock query —
not that PostgreSQL has actually queued it. Polling ``pg_locks`` from a
separate connection proves a transaction is blocked in the database before a
test releases the lock holder, so a test that passes with the lock could not
also pass without it by lucky scheduling.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import pytest
from sqlalchemy import text

from src.db.repositories.legal import LEGAL_LEDGER_LOCK_NAMESPACE

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_POLL_INTERVAL_SECONDS = 0.01


async def wait_for_advisory_lock_waiter(
    engine: AsyncEngine,
    *,
    timeout: float = 5.0,  # noqa: ASYNC109 — polls with a deadline and fails the test
    namespace: int = LEGAL_LEDGER_LOCK_NAMESPACE,
) -> None:
    """Block until some transaction is waiting on an advisory lock in ``namespace``.

    Two-key advisory locks (``pg_advisory_xact_lock(int4, int4)``) appear in
    ``pg_locks`` with ``classid`` = the first key and ``objsubid = 2``; the
    filter keeps unrelated advisory locks (other namespaces, other databases)
    from satisfying the wait.

    Args:
        engine: Engine for the test database; a fresh connection is used per poll.
        timeout: Seconds before the test fails.
        namespace: First key of the two-key advisory lock.
    """
    query = text(
        "SELECT count(*) FROM pg_locks"
        " WHERE locktype = 'advisory' AND NOT granted"
        " AND objsubid = 2 AND classid = CAST(:ns AS oid)"
        " AND database = (SELECT oid FROM pg_database WHERE datname = current_database())"
    )
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while True:
        async with engine.connect() as conn:
            waiting = (await conn.execute(query, {"ns": namespace})).scalar_one()
        if waiting >= 1:
            return
        if loop.time() >= deadline:
            pytest.fail(f"No advisory-lock waiter appeared within {timeout}s")
        await asyncio.sleep(_POLL_INTERVAL_SECONDS)
