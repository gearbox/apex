"""Tests for event schemas — round-trip encoding/decoding for all payload types."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.events import (
    BalanceUpdatedPayload,
    EventEnvelope,
    EventType,
    JobProgressPayload,
    JobStatusPayload,
    SSETicketResponse,
    SystemBroadcastRequest,
    SystemNotificationPayload,
)
from src.core.enums import NotificationLevel

_encoder = msgspec.json.Encoder()
_decoder = msgspec.json.Decoder(EventEnvelope)


def _make_envelope(event_type: EventType, payload: object) -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime(2026, 3, 17, 12, 0, 0, tzinfo=UTC),
        event_id="test-event-id",
    )


class TestJobStatusPayload:
    def test_roundtrip(self) -> None:
        job_id = uuid4()
        original = JobStatusPayload(
            job_id=job_id,
            status="completed",
            previous_status="running",
            generation_type="t2v",
            provider="grok",
        )
        encoded = _encoder.encode(original)
        decoded = msgspec.json.decode(encoded, type=JobStatusPayload)
        assert decoded.job_id == original.job_id
        assert decoded.status == original.status
        assert decoded.previous_status == original.previous_status
        assert decoded.generation_type == original.generation_type
        assert decoded.provider == original.provider

    def test_in_envelope(self) -> None:
        job_id = uuid4()
        payload = JobStatusPayload(
            job_id=job_id,
            status="completed",
            previous_status="none",
            generation_type="t2i",
            provider="aisha",
        )
        envelope = _make_envelope(EventType.JOB_STATUS_CHANGED, payload)
        wire = _encoder.encode(envelope)
        decoded_envelope = _decoder.decode(wire)
        assert decoded_envelope.event_type == EventType.JOB_STATUS_CHANGED
        assert decoded_envelope.event_id == "test-event-id"
        # payload is msgspec.Raw — bytes of the inner JSON
        inner = msgspec.json.decode(bytes(decoded_envelope.payload), type=JobStatusPayload)
        assert inner.job_id == job_id
        assert inner.status == "completed"


class TestJobProgressPayload:
    def test_roundtrip(self) -> None:
        job_id = uuid4()
        original = JobProgressPayload(job_id=job_id, progress_pct=42, generation_type="t2v")
        encoded = _encoder.encode(original)
        decoded = msgspec.json.decode(encoded, type=JobProgressPayload)
        assert decoded.job_id == job_id
        assert decoded.progress_pct == 42

    def test_in_envelope(self) -> None:
        job_id = uuid4()
        payload = JobProgressPayload(job_id=job_id, progress_pct=75, generation_type="i2v")
        envelope = _make_envelope(EventType.JOB_PROGRESS, payload)
        wire = _encoder.encode(envelope)
        decoded_envelope = _decoder.decode(wire)
        assert decoded_envelope.event_type == EventType.JOB_PROGRESS
        inner = msgspec.json.decode(bytes(decoded_envelope.payload), type=JobProgressPayload)
        assert inner.progress_pct == 75


class TestBalanceUpdatedPayload:
    def test_roundtrip(self) -> None:
        account_id = uuid4()
        original = BalanceUpdatedPayload(
            account_id=account_id,
            balance=5000,
            delta=-100,
            transaction_type="debit",
        )
        encoded = _encoder.encode(original)
        decoded = msgspec.json.decode(encoded, type=BalanceUpdatedPayload)
        assert decoded.account_id == account_id
        assert decoded.balance == 5000
        assert decoded.delta == -100
        assert decoded.transaction_type == "debit"

    def test_in_envelope(self) -> None:
        account_id = uuid4()
        payload = BalanceUpdatedPayload(
            account_id=account_id,
            balance=9900,
            delta=1000,
            transaction_type="credit",
        )
        envelope = _make_envelope(EventType.BALANCE_UPDATED, payload)
        wire = _encoder.encode(envelope)
        decoded_envelope = _decoder.decode(wire)
        assert decoded_envelope.event_type == EventType.BALANCE_UPDATED
        inner = msgspec.json.decode(bytes(decoded_envelope.payload), type=BalanceUpdatedPayload)
        assert inner.balance == 9900


class TestSystemNotificationPayload:
    def test_roundtrip_no_expires(self) -> None:
        original = SystemNotificationPayload(
            level=NotificationLevel.INFO, title="Hello", message="World"
        )
        encoded = _encoder.encode(original)
        decoded = msgspec.json.decode(encoded, type=SystemNotificationPayload)
        assert decoded.level == "info"
        assert decoded.title == "Hello"
        assert decoded.expires_at is None

    def test_roundtrip_with_expires(self) -> None:
        expires = datetime(2026, 12, 31, 23, 59, 59, tzinfo=UTC)
        original = SystemNotificationPayload(
            level=NotificationLevel.CRITICAL,
            title="Down",
            message="Maintenance",
            expires_at=expires,
        )
        encoded = _encoder.encode(original)
        decoded = msgspec.json.decode(encoded, type=SystemNotificationPayload)
        assert decoded.expires_at is not None

    def test_in_envelope(self) -> None:
        payload = SystemNotificationPayload(
            level=NotificationLevel.WARNING, title="Notice", message="Some warning"
        )
        envelope = _make_envelope(EventType.SYSTEM_NOTIFICATION, payload)
        wire = _encoder.encode(envelope)
        decoded_envelope = _decoder.decode(wire)
        assert decoded_envelope.event_type == EventType.SYSTEM_NOTIFICATION
        inner = msgspec.json.decode(bytes(decoded_envelope.payload), type=SystemNotificationPayload)
        assert inner.level == "warning"
        assert inner.title == "Notice"


class TestSSETicketResponse:
    def test_roundtrip(self) -> None:
        original = SSETicketResponse(ticket="abc123")
        encoded = _encoder.encode(original)
        decoded = msgspec.json.decode(encoded, type=SSETicketResponse)
        assert decoded.ticket == "abc123"


class TestSystemBroadcastRequest:
    def test_roundtrip(self) -> None:
        original = SystemBroadcastRequest(level=NotificationLevel.INFO, title="T", message="M")
        encoded = _encoder.encode(original)
        decoded = msgspec.json.decode(encoded, type=SystemBroadcastRequest)
        assert decoded.level == "info"
        assert decoded.expires_at is None

    def test_forbid_unknown_fields(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"level":"info","title":"T","message":"M","unknown_field":1}',
                type=SystemBroadcastRequest,
            )
