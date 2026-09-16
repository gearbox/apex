"""Integration coverage for S1 (round-2 remediation): the operation-events trust
boundary redacts node-supplied free text before it reaches a persisted row.

Mirrors test_gpu_session_operation_event_delivery.py's pattern (route handler
called directly, real Postgres) but asserts on the actual committed row content
rather than only the publish decision — the property under test is specifically
that the callback token embedded in a node's failure text never survives to disk.
"""

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
from src.core.enums import GpuSessionStatus, OperationKind, OperationStatus
from src.core.uid import new_id
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User
from src.db.repositories.gpu_session_command import GpuSessionCommandRepository
from src.db.repositories.gpu_session_operation import GpuSessionOperationRepository

_CALLBACK_TOKEN = "operation-redaction-token"
_LEAKED_SECRET = "leak-me-this-callback-token"
_TOKENIZED_URL = (
    "https://apex.test/v1/provisioning/scripts/comfyui/v1.0.0"
    f"?session=11111111-1111-1111-1111-111111111111&token={_LEAKED_SECRET}"
)


def _terminal_event(*, session_id, operation_id) -> OperationEventBody:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    return OperationEventBody(
        schema_version=2,
        event_id="terminal-failure-event",
        session_id=session_id,
        operation_id=operation_id,
        operation_kind=OperationKind.session_bootstrap,
        batch=None,
        sequence=0,
        target=None,
        status=OperationStatus.failed,
        phase=None,
        started_at=now,
        ts=now,
        elapsed_seconds=12.0,
        phase_elapsed_seconds=None,
        progress={"work": {"last_command": f"curl {_TOKENIZED_URL}"}},
        plan=None,
        summary={"final_error": f"failed: {_TOKENIZED_URL}"},
        message=f"Failed to download script from {_TOKENIZED_URL}: HTTP Error 404",
        error=f"curl {_TOKENIZED_URL} failed",
    )


async def test_terminal_failure_event_with_a_tokenized_url_is_redacted_on_disk(
    db_engine,  # type: ignore[no-untyped-def]
) -> None:
    """Read the row back from Postgres — by hand, not via the service's own
    return value — and assert the token is absent, per the staging verification
    checklist for S1."""
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"operation-redaction-{uuid4().hex}@example.com",
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
    command_id = new_id()

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
                kind=OperationKind.bundle_provision,
                command_id=command_id,
            )
            await GpuSessionCommandRepository(db).create(
                id=command_id,
                session_id=gpu_session.id,
                product_id=gpu_session.product_id,
                operation_id=operation_id,
                kind=OperationKind.bundle_provision,
                payload={},
            )

        controller = object.__new__(InternalGpuSessionController)
        settings = MagicMock()
        settings.github_content_token = ""
        settings.hf_token = ""
        settings.civitai_api_token = ""
        receiver = OperationEventService(settings=settings)
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {_CALLBACK_TOKEN}"}
        event_bus = AsyncMock()

        async with session_factory() as db:
            response = await InternalGpuSessionController.operation_event.fn(
                controller,
                session_id=gpu_session.id,
                operation_id=operation_id,
                request=request,
                data=_terminal_event(session_id=gpu_session.id, operation_id=operation_id),
                operation_event_service=receiver,
                session=db,
                event_bus=event_bus,
            )
        assert response.status_code == 200

        # Read back from Postgres by hand — a fresh session, not the one the
        # write went through.
        async with session_factory() as db:
            operation_repo = GpuSessionOperationRepository(db)
            operation = await operation_repo.get(operation_id)
            assert operation is not None
            assert _LEAKED_SECRET not in (operation.message or "")
            assert _LEAKED_SECRET not in (operation.error or "")
            assert _LEAKED_SECRET not in str(operation.progress)
            assert _LEAKED_SECRET not in str(operation.summary)
            assert "?" not in (operation.message or "")

            command = await GpuSessionCommandRepository(db).get_by_operation(operation_id)
            assert command is not None
            assert _LEAKED_SECRET not in (command.error or "")

        # And the published SSE payload — the third boundary this must cover.
        event_bus.publish.assert_awaited_once()
        published_payload = event_bus.publish.await_args.kwargs["payload"]
        assert _LEAKED_SECRET not in str(published_payload)
    finally:
        async with session_factory() as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()


def _deeply_nested(depth: int) -> dict[str, object]:
    value: dict[str, object] = {"leaf": "bottom"}
    for _ in range(depth):
        value = {"nested": value}
    return value


def _deep_summary_event(*, session_id, operation_id, depth: int) -> OperationEventBody:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    return OperationEventBody(
        schema_version=2,
        event_id="deep-summary-event",
        session_id=session_id,
        operation_id=operation_id,
        operation_kind=OperationKind.session_bootstrap,
        batch=None,
        sequence=0,
        target=None,
        status=OperationStatus.running,
        phase=None,
        started_at=now,
        ts=now,
        elapsed_seconds=1.0,
        phase_elapsed_seconds=None,
        progress=None,
        plan=None,
        summary=_deeply_nested(depth),
        message="tick",
        error=None,
    )


async def test_deeply_nested_summary_is_applied_not_500(
    db_engine,  # type: ignore[no-untyped-def]
) -> None:
    """T2, round-3 remediation: a node-supplied summary/plan/progress body can be
    arbitrarily deep. Before the depth bound, the redactor's unbounded recursive
    walk raised RecursionError on this every-tick hot path, turning a deeply
    nested (malicious or merely buggy) payload into a 500. It must now apply
    the event and persist a bounded, truncated summary instead."""
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"operation-redaction-deep-{uuid4().hex}@example.com",
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
                kind=OperationKind.bundle_provision,
            )

        controller = object.__new__(InternalGpuSessionController)
        settings = MagicMock()
        settings.github_content_token = ""
        settings.hf_token = ""
        settings.civitai_api_token = ""
        receiver = OperationEventService(settings=settings)
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {_CALLBACK_TOKEN}"}
        event_bus = AsyncMock()

        async with session_factory() as db:
            response = await InternalGpuSessionController.operation_event.fn(
                controller,
                session_id=gpu_session.id,
                operation_id=operation_id,
                request=request,
                data=_deep_summary_event(
                    session_id=gpu_session.id, operation_id=operation_id, depth=2000
                ),
                operation_event_service=receiver,
                session=db,
                event_bus=event_bus,
            )

        # Never a 500 — either applied (200) or a deliberate rejection, never
        # an unhandled RecursionError bubbling out of the handler.
        assert response.status_code == 200

        async with session_factory() as db:
            operation = await GpuSessionOperationRepository(db).get(operation_id)
            assert operation is not None
            assert "[TRUNCATED: max depth]" in str(operation.summary)
    finally:
        async with session_factory() as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()


def _malformed_url_event(*, session_id, operation_id) -> OperationEventBody:  # type: ignore[no-untyped-def]
    now = datetime.now(UTC)
    return OperationEventBody(
        schema_version=2,
        event_id="malformed-url-event",
        session_id=session_id,
        operation_id=operation_id,
        operation_kind=OperationKind.session_bootstrap,
        batch=None,
        sequence=0,
        target=None,
        status=OperationStatus.failed,
        phase=None,
        started_at=now,
        ts=now,
        elapsed_seconds=1.0,
        phase_elapsed_seconds=None,
        progress=None,
        plan=None,
        summary=None,
        message="see http://[ for details",
        error="script fetch failed: http://[",
    )


async def test_malformed_url_in_message_is_applied_not_500(
    db_engine,  # type: ignore[no-untyped-def]
) -> None:
    """U2, round-4: `urlsplit` raises `ValueError` on shapes like `http://[`,
    which ordinary node text (a wrapped/truncated URL, bracketed IPv6) can
    produce without anyone trying. Before this was guarded, that shape 500'd
    every telemetry POST for the affected session, wedging its progress
    reporting entirely. It must now apply the event and persist the redacted
    (not crashed-on) text instead."""
    session_factory = async_sessionmaker(bind=db_engine, expire_on_commit=False)
    user = User(
        id=uuid4(),
        email=f"operation-redaction-malformed-url-{uuid4().hex}@example.com",
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
                kind=OperationKind.bundle_provision,
            )

        controller = object.__new__(InternalGpuSessionController)
        settings = MagicMock()
        settings.github_content_token = ""
        settings.hf_token = ""
        settings.civitai_api_token = ""
        receiver = OperationEventService(settings=settings)
        request = MagicMock()
        request.headers = {"Authorization": f"Bearer {_CALLBACK_TOKEN}"}
        event_bus = AsyncMock()

        async with session_factory() as db:
            response = await InternalGpuSessionController.operation_event.fn(
                controller,
                session_id=gpu_session.id,
                operation_id=operation_id,
                request=request,
                data=_malformed_url_event(session_id=gpu_session.id, operation_id=operation_id),
                operation_event_service=receiver,
                session=db,
                event_bus=event_bus,
            )

        assert response.status_code == 200

        async with session_factory() as db:
            operation = await GpuSessionOperationRepository(db).get(operation_id)
            assert operation is not None
            assert operation.message == "see [REDACTED] for details"
            assert operation.error == "script fetch failed: [REDACTED]"
    finally:
        async with session_factory() as db:
            await db.execute(delete(User).where(User.id == user.id))
            await db.commit()
