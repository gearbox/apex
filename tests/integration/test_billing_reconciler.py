"""Integration tests for billing reconciler repository methods against real PostgreSQL.

These tests exercise the SQL logic the BillingReconcilerWorker depends on
directly, because the worker creates its own DB sessions via session_factory
which cannot participate in the SAVEPOINT-based test transaction.

Pattern mirrors test_gpu_session_reconciler.py.
"""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session import GpuSessionRepository

UserFactory = Callable[..., Coroutine[Any, Any, User]]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _create_stopped_session(
    db_session: AsyncSession,
    *,
    user_id: UUID,
    stopped_at: datetime,
    billing_finalized_at: datetime | None = None,
    billing_finalization_attempts: int = 0,
) -> GpuSession:
    """Insert a stopped GpuSession with the given billing state."""
    session = GpuSession(
        id=uuid4(),
        user_id=user_id,
        product_id="vex",
        bundle_name="wan_2.2_i2v",
        model_type="aisha-image",
        status=GpuSessionStatus.stopped.value,
        stopped_at=stopped_at,
        billing_finalized_at=billing_finalized_at,
        billing_finalization_attempts=billing_finalization_attempts,
    )
    db_session.add(session)
    await db_session.flush()
    return session


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_list_pending_billing_finalization_returns_stuck_session(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> None:
    """Sessions stopped before the grace cutoff with billing_finalized_at NULL appear."""
    user = await make_user(email=f"reconciler-{uuid4().hex[:6]}@example.com")
    old_stopped_at = datetime.now(UTC) - timedelta(hours=2)
    grace_cutoff = datetime.now(UTC) - timedelta(minutes=2)

    session = await _create_stopped_session(
        db_session,
        user_id=user.id,
        stopped_at=old_stopped_at,
    )

    repo = GpuSessionRepository(db_session)
    results = await repo.list_pending_billing_finalization(
        grace_cutoff=grace_cutoff,
        limit=10,
    )

    ids = [r.id for r in results]
    assert session.id in ids


async def test_list_pending_billing_finalization_excludes_finalized(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> None:
    """Sessions with billing_finalized_at already set are excluded."""
    user = await make_user(email=f"reconciler-fin-{uuid4().hex[:6]}@example.com")
    old_stopped_at = datetime.now(UTC) - timedelta(hours=2)
    grace_cutoff = datetime.now(UTC) - timedelta(minutes=2)

    finalized = await _create_stopped_session(
        db_session,
        user_id=user.id,
        stopped_at=old_stopped_at,
        billing_finalized_at=datetime.now(UTC),
    )
    pending = await _create_stopped_session(
        db_session,
        user_id=user.id,
        stopped_at=old_stopped_at,
    )

    repo = GpuSessionRepository(db_session)
    results = await repo.list_pending_billing_finalization(
        grace_cutoff=grace_cutoff,
        limit=10,
    )

    ids = [r.id for r in results]
    assert pending.id in ids
    assert finalized.id not in ids


async def test_list_pending_billing_finalization_respects_grace_period(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> None:
    """Sessions stopped after the grace cutoff are skipped."""
    user = await make_user(email=f"reconciler-grace-{uuid4().hex[:6]}@example.com")
    # Stopped just 30 seconds ago — within a 2-minute grace window
    recent_stopped_at = datetime.now(UTC) - timedelta(seconds=30)
    grace_cutoff = datetime.now(UTC) - timedelta(minutes=2)

    session = await _create_stopped_session(
        db_session,
        user_id=user.id,
        stopped_at=recent_stopped_at,
    )

    repo = GpuSessionRepository(db_session)
    results = await repo.list_pending_billing_finalization(
        grace_cutoff=grace_cutoff,
        limit=10,
    )

    ids = [r.id for r in results]
    assert session.id not in ids


async def test_list_pending_billing_finalization_limit_caps_results(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> None:
    """The limit parameter caps the number of returned rows."""
    user = await make_user(email=f"reconciler-limit-{uuid4().hex[:6]}@example.com")
    old_stopped_at = datetime.now(UTC) - timedelta(hours=2)
    grace_cutoff = datetime.now(UTC) - timedelta(minutes=2)

    for _ in range(5):
        await _create_stopped_session(
            db_session,
            user_id=user.id,
            stopped_at=old_stopped_at,
        )

    repo = GpuSessionRepository(db_session)
    results = await repo.list_pending_billing_finalization(
        grace_cutoff=grace_cutoff,
        limit=3,
    )

    assert len(results) == 3


async def test_increment_billing_finalization_attempts_bumps_counter(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> None:
    """increment_billing_finalization_attempts returns the new value and persists it."""
    user = await make_user(email=f"reconciler-bump-{uuid4().hex[:6]}@example.com")
    old_stopped_at = datetime.now(UTC) - timedelta(hours=2)

    session = await _create_stopped_session(
        db_session,
        user_id=user.id,
        stopped_at=old_stopped_at,
        billing_finalization_attempts=3,
    )

    repo = GpuSessionRepository(db_session)
    new_count = await repo.increment_billing_finalization_attempts(session.id)

    assert new_count == 4

    # Verify the value is persisted (re-fetch via the same session)
    await db_session.refresh(session)
    assert session.billing_finalization_attempts == 4


async def test_increment_billing_finalization_attempts_starts_from_zero(
    db_session: AsyncSession,
    make_user: UserFactory,
) -> None:
    """Default value of 0 increments to 1 on first bump."""
    user = await make_user(email=f"reconciler-zero-{uuid4().hex[:6]}@example.com")
    old_stopped_at = datetime.now(UTC) - timedelta(hours=2)

    session = await _create_stopped_session(
        db_session,
        user_id=user.id,
        stopped_at=old_stopped_at,
    )
    assert session.billing_finalization_attempts == 0

    repo = GpuSessionRepository(db_session)
    new_count = await repo.increment_billing_finalization_attempts(session.id)

    assert new_count == 1

    await db_session.refresh(session)
    assert session.billing_finalization_attempts == 1
