"""Integration coverage for post-commit operation-event delivery decisions."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

from sqlalchemy import delete
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.api.routes.internal_gpu_session import InternalGpuSessionController
from src.api.schemas.gpu_session import OperationEventBody
from src.api.services.gpu_session.operation_event_service import OperationEventService
from src.core.enums import GpuSessionStatus, OperationKind, OperationStatus, ProvisioningPhase
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

_CALLBACK_TOKEN = "operation-event-delivery-token"


def _event(*, session_id, operation_id, event_id: str) -> OperationEventBody:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    return OperationEventBody(
        schema_version=2,
        event_id=event_id,
        session_id=session_id,
        operation_id=operation_id,
        operation_kind=OperationKind.session_bootstrap,
        batch=None,
        sequence=0,
        target=None,
        status=OperationStatus.running,
        phase=ProvisioningPhase.preflight,
        started_at=now,
        ts=now,
        elapsed_seconds=0.0,
        phase_elapsed_seconds=None,
        progress=None,
        plan=None,
        summary=None,
        message="starting",
        error=None,
    )


async def test_stale_telemetry_does_not_emit_an_operation_update(db_engine) -> None:  # type: ignore[no-untyped-def]
    """The route publishes only a telemetry write accepted by the guarded update."""
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"operation-event-delivery-{uuid4().hex}@example.com",
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
        callback_token_hash=hashlib.sha256(_CALLBACK_TOKEN.encode()).hexdigest(),
    )
    operation_id = new_id()

    try:
        async with session_factory() as db, db.begin():
            db.add(user)
            await db.flush()
            db.add(gpu_session)
            await db.flush()
            await GpuSessionOperationRepository(db).create(
                id=operation_id,
                session_id=gpu_session.id,
                user_id=user.id,
                product_id=gpu_session.product_id,
                kind=OperationKind.session_bootstrap,
            )

        # The handler has no instance state; bypass Litestar's route-registration
        # constructor so this remains a real database/publish-path test.
        controller = object.__new__(InternalGpuSessionController)
        receiver = OperationEventService()
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {_CALLBACK_TOKEN}"}
        event_bus = AsyncMock()

        async with session_factory() as db:
            response = await InternalGpuSessionController.operation_event.fn(
                controller,
                session_id=gpu_session.id,
                operation_id=operation_id,
                request=request,
                data=_event(
                    session_id=gpu_session.id,
                    operation_id=operation_id,
                    event_id="accepted-event",
                ),
                operation_event_service=receiver,
                session=db,
                event_bus=event_bus,
            )
        assert response.status_code == 200
        event_bus.publish.assert_awaited_once()

        event_bus.reset_mock()
        async with session_factory() as db:
            response = await InternalGpuSessionController.operation_event.fn(
                controller,
                session_id=gpu_session.id,
                operation_id=operation_id,
                request=request,
                data=_event(
                    session_id=gpu_session.id,
                    operation_id=operation_id,
                    event_id="stale-event",
                ),
                operation_event_service=receiver,
                session=db,
                event_bus=event_bus,
            )
        assert response.status_code == 200
        event_bus.publish.assert_not_awaited()
    finally:
        async with session_factory() as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
