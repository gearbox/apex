"""Integration tests for guarded operation-event updates."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GpuSessionStatus, OperationKind, OperationStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine


async def test_apply_event_is_monotonic_and_terminal_once(db_session: AsyncSession) -> None:
    """Late, duplicate, and post-terminal events cannot overwrite durable state."""
    user = User(
        id=uuid4(),
        email=f"telemetry-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    db_session.add(user)
    await db_session.flush()
    db_session.add(gpu_session)
    await db_session.flush()

    repo = GpuSessionOperationRepository(db_session)
    operation = await repo.create(
        id=uuid4(),
        session_id=gpu_session.id,
        product_id="vex",
        kind=OperationKind.session_bootstrap,
    )
    now = datetime.now(UTC)
    first = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=0,
        event_id="bash-fallback-event",
        status=OperationStatus.running,
        phase="models",
        node_started_at=now,
        event_at=now,
        message="downloading",
        progress={"work": {"completed": 1, "total": 2, "unit": "files"}},
        plan={"phases": []},
        summary=None,
        error=None,
    )
    terminal = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=1,
        event_id="terminal-event",
        status=OperationStatus.succeeded,
        phase=None,
        node_started_at=now,
        event_at=now,
        message="done",
        progress=None,
        plan=None,
        summary={"total_seconds": 1.0},
        error=None,
    )
    duplicate = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=1,
        event_id="terminal-event",
        status=OperationStatus.succeeded,
        phase=None,
        node_started_at=now,
        event_at=now,
        message="changed",
        progress=None,
        plan=None,
        summary=None,
        error=None,
    )
    later = await repo.apply_event(
        operation_id=operation.id,
        session_id=gpu_session.id,
        sequence=2,
        event_id="second-terminal",
        status=OperationStatus.failed,
        phase=None,
        node_started_at=now,
        event_at=now,
        message="failed",
        progress=None,
        plan=None,
        summary=None,
        error="late failure",
    )

    await db_session.refresh(operation)
    assert first.applied is True
    assert terminal.applied is True
    assert duplicate.reason == "duplicate"
    assert later.reason == "terminal_after_terminal"
    assert operation.last_sequence == 1
    assert operation.status == OperationStatus.succeeded
    assert operation.phase is None
    assert operation.terminal_at is not None
    assert operation.node_started_at == now
    assert operation.progress == {"work": {"completed": 1, "total": 2, "unit": "files"}}


async def test_concurrent_events_apply_once_without_losing_the_higher_sequence(
    db_engine: AsyncEngine,
) -> None:
    """Two real connections race safely through the guarded UPDATE.

    The lower sequence may win first, but the higher event must still become
    the durable latest state.  NullPool supplies independent connections.
    """
    user = User(
        id=uuid4(),
        email=f"telemetry-race-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=uuid4(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    operation_id = uuid4()
    now = datetime.now(UTC)

    async with AsyncSession(bind=db_engine, expire_on_commit=False) as db:
        db.add(user)
        await db.flush()
        db.add(gpu_session)
        await db.flush()
        await GpuSessionOperationRepository(db).create(
            id=operation_id,
            session_id=gpu_session.id,
            product_id="vex",
            kind=OperationKind.session_bootstrap,
        )
        await db.commit()

    async def deliver(*, sequence: int, event_id: str) -> None:
        async with (
            AsyncSession(bind=db_engine, expire_on_commit=False) as db,
            db.begin(),
        ):
            outcome = await GpuSessionOperationRepository(db).apply_event(
                operation_id=operation_id,
                session_id=gpu_session.id,
                sequence=sequence,
                event_id=event_id,
                status=OperationStatus.succeeded if sequence == 1 else OperationStatus.running,
                phase=None if sequence == 1 else "models",
                node_started_at=now,
                event_at=now,
                message=event_id,
                progress=None,
                plan=None,
                summary={"winner": event_id} if sequence == 1 else None,
                error=None,
            )
            assert outcome.applied is True

    try:
        await asyncio.gather(
            deliver(sequence=0, event_id="concurrent-start"),
            deliver(sequence=1, event_id="concurrent-terminal"),
        )

        async with AsyncSession(bind=db_engine, expire_on_commit=False) as db:
            operation = await GpuSessionOperationRepository(db).get(operation_id)
            assert operation is not None
            assert operation.last_sequence == 1
            assert operation.last_event_id == "concurrent-terminal"
            assert operation.status == OperationStatus.succeeded
            assert operation.terminal_at is not None
    finally:
        async with AsyncSession(bind=db_engine, expire_on_commit=False) as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
