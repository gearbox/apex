"""``StudioRunEvents``: best-effort, post-commit, ids-and-codes-only publishing."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from structlog.testing import capture_logs

from src.api.dependencies import common
from src.api.schemas.events import EventType, StudioRunUpdatedPayload
from src.studio.services.run_events import StudioRunEvents

pytestmark = pytest.mark.unit


def _run(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "id": uuid4(),
        "user_id": uuid4(),
        "track_id": uuid4(),
        "kind": "plan",
        "status": "running",
        "stage": "drafting",
        "revision": 3,
        "error_code": None,
    }
    return SimpleNamespace(**{**base, **overrides})


class _Bus:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.published: list[tuple[Any, EventType, Any]] = []
        self.balance_events: list[Any] = []

    async def publish(self, *, user_id: Any, event_type: EventType, payload: object) -> None:
        if self.fail:
            raise ConnectionError("redis is down")
        self.published.append((user_id, event_type, payload))

    async def publish_balance(self, event: Any) -> None:
        self.balance_events.append(event)


async def test_the_payload_is_ids_stage_revision_and_code() -> None:
    bus = _Bus()
    run = _run(status="failed", stage=None, revision=7, error_code="plan_invalid")
    await StudioRunEvents(bus).run_updated(run)  # type: ignore[arg-type]

    ((user_id, event_type, payload),) = bus.published
    assert user_id == run.user_id
    assert event_type is EventType.STUDIO_RUN_UPDATED
    assert payload == StudioRunUpdatedPayload(
        run_id=run.id,
        track_id=run.track_id,
        kind="plan",
        status="failed",
        stage=None,
        revision=7,
        error_code="plan_invalid",
    )


async def test_the_event_type_is_the_documented_wire_name() -> None:
    assert EventType.STUDIO_RUN_UPDATED.value == "studio.run.updated"


async def test_a_publish_failure_is_swallowed_and_logged_without_text() -> None:
    run = _run()
    with capture_logs() as logs:
        await StudioRunEvents(_Bus(fail=True)).run_updated(run)  # type: ignore[arg-type]
    (entry,) = [log for log in logs if log["event"] == "studio.run.event_publish_failed"]
    assert entry["run_id"] == str(run.id)
    assert "redis" not in str(logs)


async def test_without_a_bus_nothing_happens() -> None:
    await StudioRunEvents(None).run_updated(_run())
    await StudioRunEvents(None).balance(object())  # type: ignore[arg-type]


async def test_balance_events_pass_through() -> None:
    bus = _Bus()
    marker = object()
    await StudioRunEvents(bus).balance(marker)  # type: ignore[arg-type]
    assert bus.balance_events == [marker]


def test_the_run_service_provider_refuses_to_exist_without_the_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(common._services, "studio_llm_registry", None)
    with pytest.raises(RuntimeError, match="not enabled"):
        common.get_studio_run_service(object(), object())  # type: ignore[arg-type]
