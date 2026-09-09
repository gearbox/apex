"""Integration coverage for operation frames emitted by session lifecycle cascades."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.schemas.events import EventType
from src.api.services.gpu_session import GpuSessionService, NullNodeCooldownStore
from src.core.enums import GpuSessionStatus, OperationKind, OperationStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository


async def test_terminal_session_cascade_bumps_revision_and_emits_operation_update(
    db_engine,
) -> None:  # type: ignore[no-untyped-def]
    """The D31 stop cascade publishes each operation only after its commit.

    ``publish_operation_event`` consumes the rows returned by the terminal _set_status call.
    """
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"lifecycle-operation-event-{uuid4().hex}@example.com",
        password_hash="hash",
        product_id="vex",
    )
    gpu_session = GpuSession(
        id=new_id(),
        user_id=user.id,
        product_id="vex",
        status=GpuSessionStatus.provisioning,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
    )
    operation_id = new_id()
    command_id = new_id()

    try:
        async with session_factory() as db, db.begin():
            db.add(user)
            await db.flush()
            db.add(gpu_session)
            await db.flush()
            operation = await GpuSessionOperationRepository(db).create(
                id=operation_id,
                session_id=gpu_session.id,
                user_id=user.id,
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

        event_bus = AsyncMock()
        service = GpuSessionService(
            vastai_client=AsyncMock(),
            cf_client=AsyncMock(),
            bundle_index=MagicMock(),
            session_factory=session_factory,
            settings=MagicMock(),
            billing_service=MagicMock(),
            cooldown_store=NullNodeCooldownStore(),
            event_bus=event_bus,
        )

        stopped = await service.stop_session(
            session_id=gpu_session.id,
            user_id=user.id,
            product_id=gpu_session.product_id,
        )
        assert isinstance(stopped, GpuSession)
        assert stopped.status == GpuSessionStatus.stopped

        async with session_factory() as db:
            operation = await GpuSessionOperationRepository(db).get(operation_id)
        assert operation is not None
        assert operation.status == OperationStatus.failed
        assert operation.revision == 1
        operation_frames = [
            call.kwargs["payload"]
            for call in event_bus.publish.call_args_list
            if call.kwargs["event_type"] == EventType.GPU_SESSION_OPERATION_UPDATED
        ]
        assert len(operation_frames) == 1
        assert operation_frames[0].id == operation_id
        assert operation_frames[0].revision == 1
    finally:
        async with session_factory() as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
