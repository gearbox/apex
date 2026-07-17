"""Unit tests for the pure ops-event -> Telegram-message mapping module."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.ops_events import (
    GenerationCreatedOpsPayload,
    GenerationFailedOpsPayload,
    GpuNodeStartedOpsPayload,
    HealthTransitionOpsPayload,
    OpsEventEnvelope,
    OpsEventType,
    UserRegisteredOpsPayload,
)
from src.api.services.telegram.mapping import map_ops_event
from src.core.enums import NotificationClass

_encoder = msgspec.json.Encoder()


def _envelope(event_type: OpsEventType, product_id: str, payload: object) -> OpsEventEnvelope:
    return OpsEventEnvelope(
        event_type=event_type,
        product_id=product_id,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime.now(UTC),
        event_id="evt-1",
    )


class TestUserRegistered:
    def test_maps_class_product_tag_and_id(self) -> None:
        user_id = uuid4()
        envelope = _envelope(
            OpsEventType.USER_REGISTERED, "vex", UserRegisteredOpsPayload(user_id=user_id)
        )

        notification = map_ops_event(envelope)

        assert notification is not None
        assert notification.notification_class == NotificationClass.USER_REGISTERED
        assert notification.product_id == "vex"
        assert "[vex]" in notification.text
        assert str(user_id) in notification.text


class TestGenerationCreated:
    def test_maps_provider_and_generation_type(self) -> None:
        job_id = uuid4()
        envelope = _envelope(
            OpsEventType.GENERATION_CREATED,
            "synthara",
            GenerationCreatedOpsPayload(
                job_id=job_id, user_id=uuid4(), provider="grok", generation_type="t2i"
            ),
        )

        notification = map_ops_event(envelope)

        assert notification is not None
        assert notification.notification_class == NotificationClass.GENERATION_CREATED
        assert "[synthara]" in notification.text
        assert str(job_id) in notification.text
        assert "grok/t2i" in notification.text


class TestGenerationFailed:
    def test_maps_class_and_text(self) -> None:
        job_id = uuid4()
        envelope = _envelope(
            OpsEventType.GENERATION_FAILED,
            "vex",
            GenerationFailedOpsPayload(
                job_id=job_id, user_id=uuid4(), provider="aisha", generation_type="i2v"
            ),
        )

        notification = map_ops_event(envelope)

        assert notification is not None
        assert notification.notification_class == NotificationClass.GENERATION_FAILED
        assert "Generation failed" in notification.text
        assert "aisha/i2v" in notification.text


class TestGpuNodeStarted:
    def test_maps_session_and_model_type(self) -> None:
        session_id = uuid4()
        envelope = _envelope(
            OpsEventType.GPU_NODE_STARTED,
            "vex",
            GpuNodeStartedOpsPayload(
                session_id=session_id, user_id=uuid4(), model_type="aisha-image"
            ),
        )

        notification = map_ops_event(envelope)

        assert notification is not None
        assert notification.notification_class == NotificationClass.GPU_NODE_STARTED
        assert str(session_id) in notification.text
        assert "aisha-image" in notification.text


class TestHealthTransitions:
    def test_degraded_uses_platform_tag_and_class(self) -> None:
        envelope = _envelope(
            OpsEventType.HEALTH_SUBSYSTEM_DEGRADED,
            "platform",
            HealthTransitionOpsPayload(
                subsystem="redis",
                previous_status="healthy",
                current_status="degraded",
                overall_status="degraded",
            ),
        )

        notification = map_ops_event(envelope)

        assert notification is not None
        assert notification.notification_class == NotificationClass.HEALTH_DEGRADED
        assert "[platform]" in notification.text
        assert "Health degraded" in notification.text
        assert "healthy" in notification.text
        assert "degraded" in notification.text

    def test_restored_uses_restored_class(self) -> None:
        envelope = _envelope(
            OpsEventType.HEALTH_SUBSYSTEM_RESTORED,
            "platform",
            HealthTransitionOpsPayload(
                subsystem="redis",
                previous_status="unhealthy",
                current_status="healthy",
                overall_status="healthy",
            ),
        )

        notification = map_ops_event(envelope)

        assert notification is not None
        assert notification.notification_class == NotificationClass.HEALTH_RESTORED
        assert "Health restored" in notification.text


class TestHtmlEscaping:
    @pytest.mark.parametrize(
        "raw",
        [
            "<script>alert(1)</script>",
            "a & b < c > d",
        ],
    )
    def test_injected_strings_are_escaped(self, raw: str) -> None:
        envelope = _envelope(
            OpsEventType.HEALTH_SUBSYSTEM_DEGRADED,
            raw,
            HealthTransitionOpsPayload(
                subsystem=raw,
                previous_status=raw,
                current_status=raw,
                overall_status=raw,
            ),
        )

        notification = map_ops_event(envelope)

        assert notification is not None
        assert "<script>" not in notification.text
        assert notification.text.count("&lt;") >= 1 or notification.text.count("&amp;") >= 1


class TestUnknownEventType:
    def test_unknown_event_type_maps_to_none(self) -> None:
        # Simulates a rolling-deploy scenario: a newer publisher emits an
        # event_type this dispatcher version doesn't recognize yet. msgspec
        # Struct construction (unlike decode) doesn't validate enum
        # membership, so this mirrors what a forward-incompatible decode
        # would produce.
        envelope = OpsEventEnvelope(
            event_type="ops.some_future.event_type",  # type: ignore[arg-type]
            product_id="vex",
            payload=msgspec.Raw(_encoder.encode(UserRegisteredOpsPayload(user_id=uuid4()))),
            timestamp=datetime.now(UTC),
            event_id="evt-1",
        )

        assert map_ops_event(envelope) is None
