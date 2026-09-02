"""Tests for telemetry v2 operation-event auth, routing, and projections."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

from litestar import Litestar
from litestar.di import Provide
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_404_NOT_FOUND,
)
from litestar.testing import TestClient

from src.api.routes.internal_gpu_session import InternalGpuSessionController
from src.api.schemas.gpu_session import GpuSessionResponse, OperationEventBody
from src.api.services.gpu_session.operation_event_service import (
    OperationEventService,
    _validate_token,
)
from src.core.enums import GpuSessionStatus, OperationStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_operation import GpuSessionOperation
from src.db.repositories.gpu_session_operation import EventOutcome

_SESSION_REPO = "src.api.services.gpu_session.operation_event_service.GpuSessionRepository"
_OPERATION_REPO = (
    "src.api.services.gpu_session.operation_event_service.GpuSessionOperationRepository"
)
_TOKEN = "callback-token"


def _event_body(
    *,
    session_id: UUID,
    operation_id: UUID,
    sequence: int = 0,
    event_id: str = "opaque-bash-event-id",
    status: str = "running",
    phase: str | None = "preflight",
    progress: dict[str, object] | None = None,
) -> dict[str, object]:
    now = "2026-09-02T10:00:00Z"
    return {
        "schema_version": 2,
        "event_id": event_id,
        "session_id": str(session_id),
        "operation_id": str(operation_id),
        "operation_kind": "session_bootstrap",
        "batch": None,
        "sequence": sequence,
        "target": {"bundle": "qwen_rapid_aio", "bundle_version": None, "mode": "full"},
        "status": status,
        "phase": phase,
        "started_at": now,
        "ts": now,
        "elapsed_seconds": 3.0,
        "phase_elapsed_seconds": None,
        "progress": progress,
        "plan": {"phases": []},
        "summary": None,
        "message": "Starting deployment",
        "error": None,
    }


def _decode_event(**overrides: object) -> OperationEventBody:
    session_id = overrides.pop("session_id", uuid4())
    operation_id = overrides.pop("operation_id", uuid4())
    raw = _event_body(session_id=session_id, operation_id=operation_id, **overrides)  # type: ignore[arg-type]
    return __import__("msgspec").convert(raw, type=OperationEventBody)


def _gpu_session(
    *, session_id: UUID, status: GpuSessionStatus | str = GpuSessionStatus.provisioning
) -> GpuSession:
    session = GpuSession(
        id=session_id,
        user_id=uuid4(),
        product_id="vex",
        status=status,
        bundle_name="qwen_rapid_aio",
        model_type="aisha-image",
        callback_token_hash=hashlib.sha256(_TOKEN.encode()).hexdigest(),
    )
    session.last_progress_at = None
    return session


def _operation(*, operation_id: UUID, session_id: UUID) -> GpuSessionOperation:
    return GpuSessionOperation(
        id=operation_id,
        session_id=session_id,
        product_id="vex",
        kind="session_bootstrap",
        status=OperationStatus.queued,
        last_sequence=-1,
    )


def _session_factory() -> MagicMock:
    db = MagicMock()
    db.__aenter__ = AsyncMock(return_value=db)
    db.__aexit__ = AsyncMock(return_value=None)
    begin = MagicMock()
    begin.__aenter__ = AsyncMock(return_value=None)
    begin.__aexit__ = AsyncMock(return_value=None)
    db.begin.return_value = begin
    return MagicMock(return_value=db)


class TestTokenValidation:
    def test_valid_token_matches(self) -> None:
        assert _validate_token(_TOKEN, hashlib.sha256(_TOKEN.encode()).hexdigest()) is True

    def test_wrong_or_missing_token_is_rejected(self) -> None:
        stored = hashlib.sha256(_TOKEN.encode()).hexdigest()
        assert _validate_token("wrong", stored) is False
        assert _validate_token(_TOKEN, None) is False
        assert _validate_token(_TOKEN, "") is False


class TestOperationEventService:
    async def test_bootstrap_event_is_applied_and_touches_stall_projection(self) -> None:
        session_id, operation_id = uuid4(), uuid4()
        session = _gpu_session(session_id=session_id)
        session.bootstrap_operation_id = operation_id
        event = _decode_event(session_id=session_id, operation_id=operation_id)
        service = OperationEventService(_session_factory())  # type: ignore[arg-type]

        with patch(_SESSION_REPO) as SessionRepo, patch(_OPERATION_REPO) as OperationRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = session
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo
            operation_repo.get.return_value = _operation(
                operation_id=operation_id, session_id=session_id
            )
            operation_repo.apply_event.return_value = EventOutcome(applied=True, reason="applied")

            authorized, status = await service.handle_event(
                session_id=session_id, bearer_token=_TOKEN, event=event
            )

        assert (authorized, status) == (True, HTTP_200_OK)
        session_repo.touch_last_progress.assert_awaited_once()

    async def test_non_bootstrap_event_never_touches_stall_projection(self) -> None:
        session_id, operation_id = uuid4(), uuid4()
        session = _gpu_session(session_id=session_id)
        session.bootstrap_operation_id = uuid4()
        event = _decode_event(session_id=session_id, operation_id=operation_id)
        service = OperationEventService(_session_factory())  # type: ignore[arg-type]

        with patch(_SESSION_REPO) as SessionRepo, patch(_OPERATION_REPO) as OperationRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = session
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo
            operation_repo.get.return_value = _operation(
                operation_id=operation_id, session_id=session_id
            )
            operation_repo.apply_event.return_value = EventOutcome(applied=True, reason="applied")

            assert await service.handle_event(
                session_id=session_id, bearer_token=_TOKEN, event=event
            ) == (True, 200)

        session_repo.touch_last_progress.assert_not_awaited()

    async def test_unknown_and_cross_session_operations_return_404(self) -> None:
        session_id, operation_id = uuid4(), uuid4()
        session = _gpu_session(session_id=session_id)
        event = _decode_event(session_id=session_id, operation_id=operation_id)
        service = OperationEventService(_session_factory())  # type: ignore[arg-type]

        with patch(_SESSION_REPO) as SessionRepo, patch(_OPERATION_REPO) as OperationRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            session_repo.get_by_id.return_value = session
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo
            operation_repo.get.return_value = None
            assert await service.handle_event(
                session_id=session_id, bearer_token=_TOKEN, event=event
            ) == (True, 404)
            operation_repo.get.return_value = _operation(
                operation_id=operation_id, session_id=uuid4()
            )
            assert await service.handle_event(
                session_id=session_id, bearer_token=_TOKEN, event=event
            ) == (True, 404)

        operation_repo.apply_event.assert_not_awaited()

    async def test_auth_and_terminal_session_do_not_write(self) -> None:
        session_id, operation_id = uuid4(), uuid4()
        event = _decode_event(session_id=session_id, operation_id=operation_id)
        service = OperationEventService(_session_factory())  # type: ignore[arg-type]

        with patch(_SESSION_REPO) as SessionRepo, patch(_OPERATION_REPO) as OperationRepo:
            session_repo = AsyncMock()
            SessionRepo.return_value = session_repo
            operation_repo = AsyncMock()
            OperationRepo.return_value = operation_repo
            session_repo.get_by_id.return_value = None
            assert await service.handle_event(
                session_id=session_id, bearer_token=_TOKEN, event=event
            ) == (False, 401)
            session_repo.get_by_id.return_value = _gpu_session(
                session_id=session_id, status=GpuSessionStatus.stopped
            )
            assert await service.handle_event(
                session_id=session_id, bearer_token=_TOKEN, event=event
            ) == (True, 200)

        operation_repo.get.assert_not_awaited()


def _stub_service(result: tuple[bool, int] = (True, 200)) -> OperationEventService:
    service = OperationEventService(MagicMock())
    service.handle_event = AsyncMock(return_value=result)  # type: ignore[method-assign]
    return service


def _app(service: OperationEventService) -> Litestar:
    return Litestar(
        route_handlers=[InternalGpuSessionController],
        dependencies={"operation_event_service": Provide(lambda: service, sync_to_thread=False)},
    )


class TestOperationEventController:
    def test_success_and_all_errors_are_json(self) -> None:
        session_id, operation_id = uuid4(), uuid4()
        path = f"/v1/internal/gpu-sessions/{session_id}/operations/{operation_id}/events"
        body = _event_body(session_id=session_id, operation_id=operation_id)
        service = _stub_service()

        with TestClient(app=_app(service)) as client:
            missing = client.post(path, json=body)
            mismatch = client.post(
                path,
                json={**body, "operation_id": str(uuid4())},
                headers={"Authorization": "Bearer valid"},
            )
            invalid_schema = client.post(
                path,
                json={**body, "schema_version": 3},
                headers={"Authorization": "Bearer valid"},
            )
            success = client.post(
                path,
                json={**body, "extra_future_field": True},
                headers={"Authorization": "Bearer valid"},
            )

        assert [
            missing.status_code,
            mismatch.status_code,
            invalid_schema.status_code,
            success.status_code,
        ] == [
            HTTP_401_UNAUTHORIZED,
            HTTP_400_BAD_REQUEST,
            HTTP_400_BAD_REQUEST,
            HTTP_200_OK,
        ]
        for response in (missing, mismatch, invalid_schema, success):
            assert response.headers["content-type"].startswith("application/json")

    def test_unknown_operation_is_404_json(self) -> None:
        session_id, operation_id = uuid4(), uuid4()
        path = f"/v1/internal/gpu-sessions/{session_id}/operations/{operation_id}/events"
        with TestClient(app=_app(_stub_service((True, 404)))) as client:
            response = client.post(
                path,
                json=_event_body(session_id=session_id, operation_id=operation_id),
                headers={"Authorization": "Bearer valid"},
            )
        assert response.status_code == HTTP_404_NOT_FOUND
        assert response.headers["content-type"].startswith("application/json")

    def test_bash_bootstrap_failure_accepts_opaque_event_id_and_null_target(self) -> None:
        """The pre-ACS EXIT trap emits this single terminal envelope shape."""
        session_id, operation_id = uuid4(), uuid4()
        path = f"/v1/internal/gpu-sessions/{session_id}/operations/{operation_id}/events"
        service = _stub_service()
        body = {
            **_event_body(session_id=session_id, operation_id=operation_id),
            "event_id": "bash-32767-12345",
            "sequence": 0,
            "target": None,
            "status": "failed",
            "phase": None,
            "error": "provisioning exited with status 1",
        }

        with TestClient(app=_app(service)) as client:
            response = client.post(path, json=body, headers={"Authorization": "Bearer valid"})

        assert response.status_code == HTTP_200_OK
        event = service.handle_event.await_args.kwargs["event"]  # type: ignore[attr-defined]
        assert event.event_id == "bash-32767-12345"
        assert event.target is None
        assert event.status == OperationStatus.failed


def test_response_projects_bootstrap_operation() -> None:
    now = datetime.now(UTC)
    session = _gpu_session(session_id=uuid4())
    session.created_at = now
    operation = _operation(operation_id=uuid4(), session_id=session.id)
    operation.status = OperationStatus.running
    operation.phase = "models"
    operation.progress = {"work": {"completed": 2, "total": 3, "unit": "files"}}

    response = GpuSessionResponse.from_model(session, bootstrap_operation=operation)

    assert response.provisioning_status == "running"
    assert response.provisioning_phase == "models"
    assert response.provisioning_progress == operation.progress
