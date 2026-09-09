"""Contract tests for the public OperationResponse projection."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import msgspec
import pytest
from structlog.testing import capture_logs

from src.api.schemas.events import EventType
from src.api.schemas.operation import OperationResponse
from src.api.services.gpu_session._events import publish_deployment_event, publish_operation_event
from src.core.enums import DeploymentStatus, ModelType, OperationStatus
from src.db.models.gpu_session_deployment import GpuSessionDeployment
from src.db.models.gpu_session_operation import GpuSessionOperation


def _operation(*, progress: object = None) -> GpuSessionOperation:
    now = datetime.now(UTC)
    return GpuSessionOperation(
        id=uuid4(),
        session_id=uuid4(),
        deployment_id=uuid4(),
        product_id="vex",
        user_id=uuid4(),
        kind="bundle_provision",
        status="running",
        phase="models",
        last_sequence=7,
        revision=7,
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
            "eta_basis": "live_throughput",
            "future_key": {"kept_in_db": True},
        }
    )

    with capture_logs() as logs:
        response = OperationResponse.from_model(operation)
    encoded = msgspec.json.encode(response)
    decoded = msgspec.json.decode(encoded)

    assert response.revision == 7
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
    assert "sequence" not in decoded
    assert not [entry for entry in logs if entry["event"] == "operation.progress.unprojectable"]


@pytest.mark.parametrize(
    ("eta_basis", "expected_eta", "warning_field"),
    [
        ("live_throughput", 12.3, None),
        (None, None, "eta_basis"),
        ("rough", None, "eta_basis"),
    ],
)
def test_operation_projection_exposes_eta_only_with_live_throughput_basis(
    eta_basis: object, expected_eta: float | None, warning_field: str | None
) -> None:
    progress: dict[str, object] = {
        "work": {"completed": 1, "total": 10, "unit": "files"},
        "eta_seconds": 12.3,
    }
    if eta_basis is not None:
        progress["eta_basis"] = eta_basis

    with capture_logs() as logs:
        response = OperationResponse.from_model(_operation(progress=progress))

    assert response.progress is not None
    assert response.progress.eta_seconds == expected_eta
    warnings = [entry for entry in logs if entry["event"] == "operation.progress.unprojectable"]
    if warning_field is None:
        assert not warnings
    else:
        assert [entry["field"] for entry in warnings] == [warning_field]


def test_operation_projection_clamps_progress_pct_without_editing_work() -> None:
    with capture_logs() as logs:
        response = OperationResponse.from_model(
            _operation(progress={"work": {"completed": 1500, "total": 1000, "unit": "bytes"}})
        )

    assert response.progress is not None
    assert response.progress.work is not None
    assert response.progress.work.completed == 1500
    assert response.progress.work.total == 1000
    assert response.progress.progress_pct == 100.0
    assert not [entry for entry in logs if entry["event"] == "operation.progress.unprojectable"]


@pytest.mark.parametrize(
    ("progress", "field"),
    [
        (
            {
                "work": {"completed": -1, "total": 10, "unit": "files"},
                "rate": {"value": 1, "unit": "bytes_per_second"},
            },
            "work",
        ),
        (
            {
                "work": {"completed": 1, "total": -10, "unit": "files"},
                "rate": {"value": 1, "unit": "bytes_per_second"},
            },
            "work",
        ),
        (
            {
                "items": {"completed": -1, "total": 10, "unit": "items"},
                "rate": {"value": 1, "unit": "bytes_per_second"},
            },
            "items",
        ),
        (
            {
                "items": {"completed": 1, "total": -10, "unit": "items"},
                "rate": {"value": 1, "unit": "bytes_per_second"},
            },
            "items",
        ),
        (
            {
                "work": {"completed": 1, "total": 10, "unit": "files"},
                "rate": {"value": -1, "unit": "bytes_per_second"},
            },
            "rate",
        ),
        (
            {
                "work": {"completed": 1, "total": 10, "unit": "files"},
                "eta_seconds": -1,
                "eta_basis": "live_throughput",
            },
            "eta_seconds",
        ),
    ],
)
def test_operation_projection_rejects_negative_measurements(
    progress: dict[str, object], field: str
) -> None:
    with capture_logs() as logs:
        response = OperationResponse.from_model(_operation(progress=progress))

    assert response.progress is not None
    assert getattr(response.progress, field if field != "eta_seconds" else "eta_seconds") is None
    assert any(
        entry["event"] == "operation.progress.unprojectable" and entry["field"] == field
        for entry in logs
    )


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


def test_operation_projection_treats_null_telemetry_fields_as_absent_without_warning() -> None:
    with capture_logs() as logs:
        response = OperationResponse.from_model(
            _operation(
                progress={
                    "work": {"completed": 2, "total": 3, "unit": "files"},
                    "items": {"completed": 1, "total": 2, "unit": "items"},
                    "rate": None,
                    "eta_seconds": None,
                    "eta_basis": None,
                }
            )
        )

    assert response.progress is not None
    assert response.progress.work is not None
    assert response.progress.items is not None
    assert response.progress.rate is None
    assert response.progress.eta_seconds is None
    assert not [entry for entry in logs if entry["event"] == "operation.progress.unprojectable"]


def test_operation_projection_tolerates_unknown_foreign_phase_without_warning_crash() -> None:
    operation = _operation()
    operation.phase = "future_phase"

    with capture_logs() as logs:
        response = OperationResponse.from_model(operation)

    assert response.phase is None
    assert any(
        entry["event"] == "operation.progress.unprojectable" and entry["field"] == "phase"
        for entry in logs
    )


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


def test_operation_response_deployment_id_is_described_in_openapi() -> None:
    """Keep the SSE cache-patching rule visible to generated-client users."""
    from litestar import Litestar, post
    from litestar.openapi.config import OpenAPIConfig

    async def handler(data):  # type: ignore[no-untyped-def]
        return data

    handler.__annotations__ = {"data": OperationResponse, "return": OperationResponse}
    app = Litestar(
        [post("/operation")(handler)],
        openapi_config=OpenAPIConfig(title="Operation schema", version="1"),
    )
    schema = app.openapi_schema.to_schema()
    deployment_id = schema["components"]["schemas"]["OperationResponse"]["properties"][
        "deployment_id"
    ]

    description = deployment_id.get("description")
    assert isinstance(description, str)
    assert description
    assert "must never" in description

    eta_seconds = schema["components"]["schemas"]["OperationProgressResponse"]["properties"][
        "eta_seconds"
    ]
    eta_description = eta_seconds.get("description")
    assert isinstance(eta_description, str)
    assert "live throughput" in eta_description


def test_operation_projection_keeps_bootstrap_target_but_not_cohort_target() -> None:
    bootstrap = _operation()
    bootstrap.kind = "session_bootstrap"
    cohort_restart = _operation()
    cohort_restart.kind = "comfyui_restart"
    cohort_restart.deployment_id = None

    assert OperationResponse.from_model(bootstrap).deployment_id is not None
    assert OperationResponse.from_model(cohort_restart).deployment_id is None


class _RecordingEventBus:
    def __init__(self) -> None:
        self.user_id: UUID | None = None
        self.event_type: EventType | None = None
        self.payload: object | None = None

    async def publish(self, *, user_id: UUID, event_type: EventType, payload: object) -> None:
        self.user_id = user_id
        self.event_type = event_type
        self.payload = payload


async def test_deployment_status_sse_payload_omits_raw_operation_telemetry() -> None:
    """Only the typed operation-update event may carry operation progress."""
    deployment = GpuSessionDeployment(
        id=uuid4(),
        session_id=uuid4(),
        user_id=uuid4(),
        product_id="vex",
        model_type=ModelType.AISHA_IMAGE,
        bundle_name="qwen_rapid_aio",
        status=DeploymentStatus.deploying,
    )
    operation = _operation(
        progress={
            "work": {"completed": 1, "total": 2, "unit": "files"},
            "opaque_aisha_field": {"must_not_escape": True},
        }
    )
    operation.session_id = deployment.session_id
    operation.deployment_id = deployment.id
    bus = _RecordingEventBus()

    await publish_deployment_event(bus, deployment, operation=operation)  # type: ignore[arg-type]

    assert bus.event_type is EventType.GPU_DEPLOYMENT_STATUS_CHANGED
    encoded = msgspec.json.encode(bus.payload)
    payload = msgspec.json.decode(encoded)
    assert payload["operation_id"] == str(operation.id)
    assert "operation_phase" not in payload
    assert "operation_progress" not in payload


async def test_operation_sse_payload_is_the_exact_rest_projection() -> None:
    operation = _operation(progress={"work": {"completed": 1, "total": 2, "unit": "files"}})
    bus = _RecordingEventBus()
    user_id = operation.user_id

    await publish_operation_event(bus, operation)  # type: ignore[arg-type]

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
