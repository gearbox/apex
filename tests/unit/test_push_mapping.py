"""Unit tests for the pure event -> push-notification mapping module."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.events import (
    BalanceUpdatedPayload,
    EventEnvelope,
    EventType,
    GpuSessionCreditWarningPayload,
    GpuSessionStatusPayload,
    JobProgressPayload,
    JobStatusPayload,
    SystemNotificationPayload,
)
from src.api.services.push_mapping import map_event_to_notification
from src.core.enums import NotificationLevel

_encoder = msgspec.json.Encoder()


def _envelope(event_type: EventType, payload: object) -> EventEnvelope:
    return EventEnvelope(
        event_type=event_type,
        payload=msgspec.Raw(_encoder.encode(payload)),
        timestamp=datetime.now(UTC),
        event_id="evt-1",
    )


# ---------------------------------------------------------------------------
# job.status_changed
# ---------------------------------------------------------------------------


class TestJobStatusChanged:
    @pytest.mark.parametrize("status", ["completed", "failed"])
    def test_terminal_status_maps_to_notification(self, status: str) -> None:
        job_id = uuid4()
        envelope = _envelope(
            EventType.JOB_STATUS_CHANGED,
            JobStatusPayload(
                job_id=job_id,
                status=status,
                previous_status="running",
                generation_type="t2v",
                provider="grok",
            ),
        )

        notification = map_event_to_notification(envelope)

        assert notification is not None
        assert notification.category == "job"
        assert notification.tag == f"job-{job_id}"
        assert notification.url == f"/app/library/groups/{job_id}"

    @pytest.mark.parametrize("status", ["pending", "queued", "running", "cancelled", "moderated"])
    def test_non_terminal_status_is_ignored(self, status: str) -> None:
        envelope = _envelope(
            EventType.JOB_STATUS_CHANGED,
            JobStatusPayload(
                job_id=uuid4(),
                status=status,
                previous_status="pending",
                generation_type="t2i",
                provider="aisha",
            ),
        )

        assert map_event_to_notification(envelope) is None

    def test_completed_and_failed_have_distinct_titles_and_levels(self) -> None:
        completed = _envelope(
            EventType.JOB_STATUS_CHANGED,
            JobStatusPayload(
                job_id=uuid4(),
                status="completed",
                previous_status="running",
                generation_type="t2i",
                provider="grok",
            ),
        )
        failed = _envelope(
            EventType.JOB_STATUS_CHANGED,
            JobStatusPayload(
                job_id=uuid4(),
                status="failed",
                previous_status="running",
                generation_type="t2i",
                provider="grok",
            ),
        )

        completed_notification = map_event_to_notification(completed)
        failed_notification = map_event_to_notification(failed)

        assert completed_notification is not None
        assert failed_notification is not None
        assert completed_notification.title != failed_notification.title
        assert completed_notification.level == "info"
        assert failed_notification.level == "warning"


# ---------------------------------------------------------------------------
# job.progress / gpu_session.status_changed — always ignored
# ---------------------------------------------------------------------------


class TestIgnoredEvents:
    def test_job_progress_is_ignored(self) -> None:
        envelope = _envelope(
            EventType.JOB_PROGRESS,
            JobProgressPayload(job_id=uuid4(), progress_pct=50, generation_type="t2v"),
        )
        assert map_event_to_notification(envelope) is None

    def test_gpu_session_status_changed_is_ignored(self) -> None:
        envelope = _envelope(
            EventType.GPU_SESSION_STATUS_CHANGED,
            GpuSessionStatusPayload(
                session_id=uuid4(),
                status="active",
                previous_status="provisioning",
                model_type="aisha-image",
            ),
        )
        assert map_event_to_notification(envelope) is None


# ---------------------------------------------------------------------------
# gpu_session.credit_warning
# ---------------------------------------------------------------------------


class TestCreditWarning:
    @pytest.mark.parametrize("level", [NotificationLevel.WARNING, NotificationLevel.CRITICAL])
    def test_all_levels_map_to_notification(self, level: NotificationLevel) -> None:
        session_id = uuid4()
        envelope = _envelope(
            EventType.GPU_SESSION_CREDIT_WARNING,
            GpuSessionCreditWarningPayload(
                session_id=session_id,
                level=level,
                minutes_remaining=15,
                terminate_at=None,
                balance=1000,
            ),
        )

        notification = map_event_to_notification(envelope)

        assert notification is not None
        assert notification.category == "gpu_credit"
        assert notification.tag == f"gpu-credit-{session_id}"
        assert notification.level == level.value
        assert "15" in notification.body


# ---------------------------------------------------------------------------
# system.notification
# ---------------------------------------------------------------------------


class TestSystemNotification:
    def test_maps_title_and_message_through(self) -> None:
        envelope = _envelope(
            EventType.SYSTEM_NOTIFICATION,
            SystemNotificationPayload(
                level=NotificationLevel.CRITICAL,
                title="Scheduled maintenance",
                message="The service will be down for 10 minutes.",
            ),
        )

        notification = map_event_to_notification(envelope)

        assert notification is not None
        assert notification.title == "Scheduled maintenance"
        assert notification.body == "The service will be down for 10 minutes."
        assert notification.category == "system"
        assert notification.level == "critical"


# ---------------------------------------------------------------------------
# balance.updated
# ---------------------------------------------------------------------------


class TestBalanceUpdated:
    @pytest.mark.parametrize("transaction_type", ["credit", "admin_adjustment"])
    def test_positive_delta_payment_or_admin_credit_maps_to_notification(
        self, transaction_type: str
    ) -> None:
        account_id = uuid4()
        envelope = _envelope(
            EventType.BALANCE_UPDATED,
            BalanceUpdatedPayload(
                account_id=account_id,
                balance=1500,
                delta=500,
                transaction_type=transaction_type,
            ),
        )

        notification = map_event_to_notification(envelope)

        assert notification is not None
        assert notification.category == "balance"
        assert notification.tag == f"balance-{account_id}"

    def test_positive_delta_refund_is_ignored(self) -> None:
        """Refund is a positive delta but is not a payment/top-up — must not push."""
        envelope = _envelope(
            EventType.BALANCE_UPDATED,
            BalanceUpdatedPayload(
                account_id=uuid4(),
                balance=1500,
                delta=500,
                transaction_type="refund",
            ),
        )

        assert map_event_to_notification(envelope) is None

    def test_debit_negative_delta_is_ignored(self) -> None:
        envelope = _envelope(
            EventType.BALANCE_UPDATED,
            BalanceUpdatedPayload(
                account_id=uuid4(),
                balance=500,
                delta=-100,
                transaction_type="debit",
            ),
        )

        assert map_event_to_notification(envelope) is None

    def test_zero_delta_is_ignored(self) -> None:
        envelope = _envelope(
            EventType.BALANCE_UPDATED,
            BalanceUpdatedPayload(
                account_id=uuid4(),
                balance=500,
                delta=0,
                transaction_type="credit",
            ),
        )

        assert map_event_to_notification(envelope) is None
