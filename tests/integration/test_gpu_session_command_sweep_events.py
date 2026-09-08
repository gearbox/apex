"""Integration coverage for operation updates emitted by the command sweep."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.schemas.events import EventType
from src.api.services.gpu_session.command_sweep_worker import GpuSessionCommandSweepWorker
from src.core.enums import CommandStatus, GpuSessionStatus, OperationKind, OperationStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_command import GpuSessionCommand
from src.db.models.user import User
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository


class _Settings:
    gpu_command_sweep_interval_seconds = 60


async def test_command_sweep_timeout_bumps_revision_and_emits_operation_update(
    db_engine,
) -> None:  # type: ignore[no-untyped-def]
    """A timeout reaches the owner with the same revision as the durable row."""
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"command-sweep-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.active,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    operation_id = new_id()
    command_id = new_id()
    now = datetime.now(UTC)

    try:
        async with session_factory() as db, db.begin():
            db.add(user)
            await db.flush()
            db.add(gpu_session)
            await db.flush()
            operation = await GpuSessionOperationRepository(db).create(
                id=operation_id,
                session_id=gpu_session.id,
                product_id=gpu_session.product_id,
                kind=OperationKind.bundle_provision,
                command_id=command_id,
            )
            await GpuSessionCommandRepository(db).create(
                id=command_id,
                session_id=gpu_session.id,
                product_id=gpu_session.product_id,
                operation_id=operation.id,
                kind=OperationKind.bundle_provision,
                payload={},
            )
            await db.execute(
                update(GpuSessionCommand)
                .where(GpuSessionCommand.id == command_id)
                .values(
                    status=CommandStatus.claimed,
                    claimed_at=now - timedelta(minutes=2),
                    deadline_at=now - timedelta(minutes=1),
                )
            )

        event_bus = AsyncMock()
        worker = GpuSessionCommandSweepWorker(
            session_factory=session_factory,
            settings=_Settings(),  # type: ignore[arg-type]
            event_bus=event_bus,
            redis_enabled=False,
            redis_client_factory=MagicMock(),
        )
        await worker.run_once()

        async with session_factory() as db:
            operation = await GpuSessionOperationRepository(db).get(operation_id)
        assert operation is not None
        assert operation.status == OperationStatus.failed
        assert operation.revision == 1
        event_bus.publish.assert_awaited_once()
        publish_call = event_bus.publish.await_args
        assert publish_call.kwargs["user_id"] == user.id
        assert publish_call.kwargs["event_type"] == EventType.GPU_SESSION_OPERATION_UPDATED
        assert publish_call.kwargs["payload"].status == OperationStatus.failed
        assert publish_call.kwargs["payload"].revision == 1
    finally:
        async with session_factory() as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
