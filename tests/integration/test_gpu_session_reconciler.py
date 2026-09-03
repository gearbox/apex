"""Integration tests for GpuSessionReconciler with real PostgreSQL."""

from __future__ import annotations

from collections.abc import Callable, Coroutine
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import httpx
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.api.services.health.checkers.gpu_session import GpuSessionReconciler
from src.core.enums import DeploymentStatus, GpuSessionStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.user import User
from src.db.repositories.gpu_session_deployment import GpuSessionDeploymentRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

UserFactory = Callable[..., Coroutine[Any, Any, User]]


async def _create_gpu_session(
    db_session: AsyncSession,
    *,
    user_id: object = None,
    status: str = GpuSessionStatus.active.value,
    tunnel_hostname: str = "gpu-test01.gpu-domain.com",
) -> GpuSession:
    """Insert a GpuSession directly via the test session."""
    gpu_session = GpuSession(
        id=new_id(),
        user_id=uuid4() if user_id is None else user_id,
        product_id="vex",
        bundle_name="test_bundle",
        model_type="aisha-image",
        status=status,
        tunnel_hostname=tunnel_hostname,
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
        gpu_sess = await _create_gpu_session(
            db_session, user_id=user.id, tunnel_hostname="gpu-test99.gpu-domain.com"
        )

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

    async def test_recovery_preserves_routable_deployment(self, db_engine: AsyncEngine) -> None:
        """The health checker's active → stale → active cycle keeps routing intact.

        The checker commits through its own session factory, so this deliberately
        uses committed rows rather than the SAVEPOINT-backed ``db_session`` fixture.
        """
        user = User(
            id=new_id(),
            email=f"gpu-recovery-{uuid4().hex}@example.com",
            password_hash="hash",
            product_id="vex",
        )
        gpu_session = GpuSession(
            id=new_id(),
            user_id=user.id,
            product_id="vex",
            status=GpuSessionStatus.active,
            bundle_name="test_bundle",
            model_type="aisha-image",
            tunnel_hostname="gpu-test-recovery.gpu-domain.com",
            started_at=datetime(2026, 1, 1, tzinfo=UTC),
        )
        deployment = GpuSessionDeployment(
            id=new_id(),
            session_id=gpu_session.id,
            user_id=user.id,
            product_id="vex",
            model_type="aisha-image",
            bundle_name="test_bundle",
            status=DeploymentStatus.active,
            is_primary=True,
        )
        session_factory = async_sessionmaker(db_engine, expire_on_commit=False)

        async def _routable() -> bool:
            async with session_factory() as db:
                return (
                    await GpuSessionDeploymentRepository(db).get_routable(
                        user.id, "vex", "aisha-image"
                    )
                    is not None
                )

        try:
            async with session_factory() as db:
                # These models have no ORM relationships, so flush in foreign-key
                # order rather than relying on unit-of-work dependency sorting.
                db.add(user)
                await db.flush()
                db.add(gpu_session)
                await db.flush()
                db.add(deployment)
                await db.commit()

            async with httpx.AsyncClient() as http_client:
                checker = GpuSessionReconciler(
                    session_factory,
                    http_client,
                    tunnel_domain="gpu-domain.com",
                    tunnel_prefix="gpu-",
                    grace_seconds=120,
                )
                assert await _routable()

                await checker._mark_stale([gpu_session.id], now=datetime(2026, 1, 2, tzinfo=UTC))
                # The active deployment remains intact while its stale parent
                # correctly prevents routing.
                assert not await _routable()

                await checker._clear_stale_batch(
                    [gpu_session.id], now=datetime(2026, 1, 3, tzinfo=UTC)
                )

            assert await _routable()
        finally:
            async with session_factory() as db:
                await db.execute(delete(User).where(User.id == user.id))
                await db.commit()
