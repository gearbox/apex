"""Integration tests for GpuSessionReconciler with real PostgreSQL."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GpuSessionStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User

UserFactory = Callable[..., Coroutine[Any, Any, User]]


async def _create_gpu_session(
    db_session: AsyncSession,
    *,
    user_id: object = None,
    status: str = GpuSessionStatus.active.value,
    host: str = "10.0.0.1",
    port: int = 8188,
) -> GpuSession:
    """Insert a GpuSession directly via the test session."""
    gpu_session = GpuSession(
        id=new_id(),
        user_id=uuid4() if user_id is None else user_id,
        product_id="vex",
        status=status,
        node_host=host,
        node_port=port,
    )
    db_session.add(gpu_session)
    await db_session.flush()
    return gpu_session


class TestReconcilerIntegration:
    """Integration tests using real DB session.

    These tests exercise the SQL logic directly (UPDATE statements,
    partial index behavior, idempotency) rather than running the full
    reconciler, because the reconciler creates its own DB sessions via
    session_factory which cannot participate in the SAVEPOINT transaction.
    """

    async def test_mark_stale_persists(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """Verify the stale-marking UPDATE writes correct values to DB."""
        user = await make_user(email=f"gpu-{uuid4().hex[:8]}@example.com")
        gpu_sess = await _create_gpu_session(db_session, user_id=user.id, host="10.0.0.99")

        # Execute the same UPDATE the reconciler's _mark_stale uses
        stmt = (
            update(GpuSession)
            .where(
                GpuSession.id == gpu_sess.id,
                GpuSession.stale_detected_at.is_(None),
            )
            .values(
                stale_detected_at=datetime.now(UTC),
                status=GpuSessionStatus.stale.value,
            )
        )
        await db_session.execute(stmt)
        await db_session.flush()

        result = await db_session.execute(select(GpuSession).where(GpuSession.id == gpu_sess.id))
        updated = result.scalar_one()
        assert updated.status == GpuSessionStatus.stale.value
        assert updated.stale_detected_at is not None

    async def test_mark_stale_idempotent(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """Second stale-mark UPDATE must not overwrite the original detection time."""
        user = await make_user(email=f"gpu-{uuid4().hex[:8]}@example.com")
        original_time = datetime(2026, 1, 1, tzinfo=UTC)
        gpu_sess = await _create_gpu_session(db_session, user_id=user.id)

        # First mark
        await db_session.execute(
            update(GpuSession)
            .where(GpuSession.id == gpu_sess.id, GpuSession.stale_detected_at.is_(None))
            .values(stale_detected_at=original_time, status=GpuSessionStatus.stale.value)
        )
        await db_session.flush()

        # Second mark — WHERE stale_detected_at IS NULL prevents this from changing anything
        await db_session.execute(
            update(GpuSession)
            .where(GpuSession.id == gpu_sess.id, GpuSession.stale_detected_at.is_(None))
            .values(
                stale_detected_at=datetime(2026, 6, 1, tzinfo=UTC),
                status=GpuSessionStatus.stale.value,
            )
        )
        await db_session.flush()

        result = await db_session.execute(select(GpuSession).where(GpuSession.id == gpu_sess.id))
        final = result.scalar_one()
        assert final.stale_detected_at == original_time

    async def test_clear_stale_restores_active(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """Verify the clear-stale UPDATE transitions status back to active."""
        user = await make_user(email=f"gpu-{uuid4().hex[:8]}@example.com")
        gpu_sess = await _create_gpu_session(
            db_session, user_id=user.id, status=GpuSessionStatus.stale.value
        )

        # Simulate the _mark_stale having run previously
        await db_session.execute(
            update(GpuSession)
            .where(GpuSession.id == gpu_sess.id)
            .values(stale_detected_at=datetime(2026, 1, 1, tzinfo=UTC))
        )
        await db_session.flush()

        # Execute the same UPDATE the reconciler's _clear_stale uses
        await db_session.execute(
            update(GpuSession)
            .where(GpuSession.id == gpu_sess.id)
            .values(
                stale_detected_at=None,
                stale_notified=False,
                status=GpuSessionStatus.active.value,
            )
        )
        await db_session.flush()

        result = await db_session.execute(select(GpuSession).where(GpuSession.id == gpu_sess.id))
        recovered = result.scalar_one()
        assert recovered.status == GpuSessionStatus.active.value
        assert recovered.stale_detected_at is None
        assert recovered.stale_notified is False
