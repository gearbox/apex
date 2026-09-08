"""Contract tests for the public OperationResponse projection."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import msgspec

from src.api.schemas.events import EventType
from src.api.schemas.operation import OperationResponse
from src.api.services.gpu_session._events import publish_operation_event
from src.core.enums import OperationStatus
from src.db.models.gpu_session_operation import GpuSessionOperation


def _operation(*, progress: object = None) -> GpuSessionOperation:
    now = datetime.now(UTC)
    return GpuSessionOperation(
        id=uuid4(),
        session_id=uuid4(),
        deployment_id=uuid4(),
        product_id="vex",
        kind="bundle_provision",
        status="running",
        phase="models",
        last_sequence=7,
        target_bundle="wan",
        target_bundle_version="20260908-01",
        target_mode="additive",
        progress=progress,
        plan={"private": True},
        summary={"private": True},
        created_at=now,
        updated_at=now,
    )


def test_operation_projection_computes_progress_and_omits_private_telemetry() -> None:
    operation = _operation(
        progress={
            "work": {"completed": 2, "total": 3, "unit": "files"},
            "items": {"completed": 4, "total": None, "unit": "items"},
            "rate": {"value": 128.5, "unit": "bytes_per_second"},
            "eta_seconds": 12.25,
            "future_key": {"kept_in_db": True},
        }
    )

    response = OperationResponse.from_model(operation)
    encoded = msgspec.json.encode(response)
    decoded = msgspec.json.decode(encoded)

    assert response.sequence == 7
    assert response.progress is not None
    assert response.progress.progress_pct == 66.7
    assert response.progress.work is not None
    assert response.progress.work.completed == 2
    assert response.progress.items is not None
    assert response.progress.rate is not None
    assert response.progress.eta_seconds == 12.25
    assert "future_key" not in decoded["progress"]
    assert "plan" not in decoded
    assert "summary" not in decoded


def test_operation_projection_keeps_progress_pct_none_without_positive_work_total() -> None:
    for work in (
        {"completed": 1, "total": 0, "unit": "files"},
        {"completed": 1, "total": None, "unit": "files"},
        None,
    ):
        response = OperationResponse.from_model(_operation(progress={"work": work}))
        assert response.progress is None or response.progress.progress_pct is None


def test_operation_projection_tolerates_malformed_foreign_json() -> None:
    response = OperationResponse.from_model(
        _operation(progress={"work": ["not", "an", "object"], "rate": "bad"})
    )
    assert response.progress is None


def test_operation_projection_requires_complete_target_and_wraps_errors() -> None:
    operation = _operation()
    operation.target_bundle = None
    operation.target_bundle_version = "orphaned-version"
    operation.target_mode = "additive"
    operation.error = "node failed"

    response = OperationResponse.from_model(operation)

    assert response.target is None
    assert response.error is not None
    assert response.error.message == "node failed"


class _RecordingEventBus:
    def __init__(self) -> None:
        self.user_id: UUID | None = None
        self.event_type: EventType | None = None
        self.payload: object | None = None

    async def publish(self, *, user_id: UUID, event_type: EventType, payload: object) -> None:
        self.user_id = user_id
        self.event_type = event_type
        self.payload = payload


async def test_operation_sse_payload_is_the_exact_rest_projection() -> None:
    operation = _operation(progress={"work": {"completed": 1, "total": 2, "unit": "files"}})
    bus = _RecordingEventBus()
    user_id = uuid4()

    await publish_operation_event(bus, operation, user_id=user_id)  # type: ignore[arg-type]

    assert bus.user_id == user_id
    assert bus.event_type is EventType.GPU_SESSION_OPERATION_UPDATED
    assert isinstance(bus.payload, OperationResponse)
    assert msgspec.json.encode(bus.payload) == msgspec.json.encode(
        OperationResponse.from_model(operation)
    )
    assert bus.payload.status is OperationStatus.running


async def test_operation_event_publish_is_a_two_argument_noop_without_sse() -> None:
    """Redis-less deployments may call the helper without resolving an owner."""
    await publish_operation_event(None, _operation())
