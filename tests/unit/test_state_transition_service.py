"""Tests for JobStateTransitionService.

Covers idempotency, output persistence, event publishing, and refund wiring.
All DB calls are mocked so no real session is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.job_state_transition import (
    GenerationOutputData,
    JobStateTransitionService,
)
from src.core.enums import JobStatus

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_job(status: str = JobStatus.QUEUED.value) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.user_id = uuid4()
    job.status = status
    job.generation_type = "t2i"
    job.provider = "aisha"
    job.debit_transaction_id = uuid4()
    return job


def _make_service(
    job: MagicMock,
    *,
    rowcount: int = 1,
    event_bus: AsyncMock | None = None,
) -> tuple[JobStateTransitionService, AsyncMock, AsyncMock]:
    session = AsyncMock()
    # session.get returns the job, session.refresh re-attaches it
    session.get.return_value = job
    session.refresh = AsyncMock()
    # Simulate the UPDATE rowcount
    execute_result = MagicMock()
    execute_result.rowcount = rowcount
    session.execute.return_value = execute_result

    billing = AsyncMock()
    svc = JobStateTransitionService(
        session=session,
        event_bus=event_bus,
        billing_service=billing,
    )
    return svc, session, billing


# ---------------------------------------------------------------------------
# transition_to_running
# ---------------------------------------------------------------------------


class TestTransitionToRunning:
    async def test_queued_to_running(self) -> None:
        job = _make_job(JobStatus.QUEUED.value)
        svc, session, _ = _make_service(job)

        await svc.transition_to_running(job.id)

        session.execute.assert_awaited_once()
        session.commit.assert_awaited_once()

    async def test_already_running_is_noop(self) -> None:
        job = _make_job(JobStatus.RUNNING.value)
        svc, session, _ = _make_service(job)

        await svc.transition_to_running(job.id)

        session.execute.assert_not_awaited()
        session.commit.assert_not_awaited()

    async def test_already_completed_is_noop(self) -> None:
        job = _make_job(JobStatus.COMPLETED.value)
        svc, session, _ = _make_service(job)

        await svc.transition_to_running(job.id)

        session.execute.assert_not_awaited()

    async def test_concurrent_update_no_double_publish(self) -> None:
        job = _make_job(JobStatus.QUEUED.value)
        bus = AsyncMock()
        svc, _session, _ = _make_service(job, rowcount=0, event_bus=bus)

        await svc.transition_to_running(job.id)

        bus.publish.assert_not_awaited()

    async def test_job_not_found_raises(self) -> None:
        session = AsyncMock()
        session.get.return_value = None
        billing = AsyncMock()
        svc = JobStateTransitionService(session=session, event_bus=None, billing_service=billing)

        with pytest.raises(ValueError, match="not found"):
            await svc.transition_to_running(uuid4())


# ---------------------------------------------------------------------------
# transition_to_completed
# ---------------------------------------------------------------------------


class TestTransitionToCompleted:
    def _make_output(self, index: int = 0) -> GenerationOutputData:
        return GenerationOutputData(
            id=uuid4(),
            storage_key=f"users/u/outputs/j/img{index}.webp",
            content_type="image/webp",
            size_bytes=12345,
            format="webp",
            output_index=index,
            expires_at=datetime.now(UTC) + timedelta(days=7),
        )

    async def test_running_to_completed(self) -> None:
        job = _make_job(JobStatus.RUNNING.value)
        svc, session, _ = _make_service(job)

        with patch.object(svc._output_repo, "create", new_callable=AsyncMock) as mock_create:
            out = self._make_output()
            await svc.transition_to_completed(job.id, outputs=[out], product_id="vex")
            mock_create.assert_awaited_once()

        session.commit.assert_awaited_once()

    async def test_already_completed_is_noop(self) -> None:
        job = _make_job(JobStatus.COMPLETED.value)
        svc, session, _ = _make_service(job)

        await svc.transition_to_completed(job.id, outputs=[], product_id="vex")

        session.execute.assert_not_awaited()

    async def test_publishes_event_after_commit(self) -> None:
        job = _make_job(JobStatus.RUNNING.value)
        bus = AsyncMock()
        svc, _, _ = _make_service(job, event_bus=bus)

        with patch.object(svc._output_repo, "create", new_callable=AsyncMock):
            await svc.transition_to_completed(job.id, outputs=[], product_id="vex")

        bus.publish.assert_awaited_once()

    async def test_publish_failure_does_not_raise(self) -> None:
        job = _make_job(JobStatus.RUNNING.value)
        bus = AsyncMock()
        bus.publish.side_effect = RuntimeError("redis down")
        svc, _, _ = _make_service(job, event_bus=bus)

        with patch.object(svc._output_repo, "create", new_callable=AsyncMock):
            await svc.transition_to_completed(job.id, outputs=[], product_id="vex")


# ---------------------------------------------------------------------------
# transition_to_failed
# ---------------------------------------------------------------------------


class TestTransitionToFailed:
    async def test_queued_to_failed(self) -> None:
        job = _make_job(JobStatus.QUEUED.value)
        svc, session, billing = _make_service(job)

        _, did = await svc.transition_to_failed(job.id, error_message="boom", product_id="vex")

        assert did is True
        session.commit.assert_awaited()
        billing.refund.assert_awaited_once()

    async def test_running_to_failed_returns_did_true(self) -> None:
        job = _make_job(JobStatus.RUNNING.value)
        svc, _, _ = _make_service(job)

        _, did = await svc.transition_to_failed(job.id, error_message="x", product_id="vex")

        assert did is True

    async def test_already_failed_returns_did_false(self) -> None:
        job = _make_job(JobStatus.FAILED.value)
        svc, session, billing = _make_service(job)

        _, did = await svc.transition_to_failed(job.id, error_message="x", product_id="vex")

        assert did is False
        session.execute.assert_not_awaited()
        billing.refund.assert_not_awaited()

    async def test_already_completed_returns_did_false(self) -> None:
        job = _make_job(JobStatus.COMPLETED.value)
        svc, session, billing = _make_service(job)

        _, did = await svc.transition_to_failed(job.id, error_message="x", product_id="vex")

        assert did is False
        session.execute.assert_not_awaited()
        billing.refund.assert_not_awaited()

    async def test_did_false_does_not_publish(self) -> None:
        job = _make_job(JobStatus.FAILED.value)
        bus = AsyncMock()
        svc, _, _ = _make_service(job, event_bus=bus)

        _, did = await svc.transition_to_failed(job.id, error_message="x", product_id="vex")

        assert did is False
        bus.publish.assert_not_awaited()

    async def test_did_false_does_not_refund(self) -> None:
        job = _make_job(JobStatus.COMPLETED.value)
        svc, _, billing = _make_service(job)

        _, did = await svc.transition_to_failed(job.id, error_message="x", product_id="vex")

        assert did is False
        billing.refund.assert_not_awaited()

    async def test_no_refund_when_flag_false(self) -> None:
        job = _make_job(JobStatus.RUNNING.value)
        svc, _, billing = _make_service(job)

        _, _ = await svc.transition_to_failed(
            job.id, error_message="x", refund=False, product_id="vex"
        )

        billing.refund.assert_not_awaited()

    async def test_refund_failure_rolls_back_the_terminal_transition(self) -> None:
        job = _make_job(JobStatus.RUNNING.value)
        svc, session, billing = _make_service(job)
        billing.refund.side_effect = RuntimeError("billing down")

        with pytest.raises(RuntimeError, match="billing down"):
            await svc.transition_to_failed(job.id, error_message="x", product_id="vex")

        session.rollback.assert_awaited()
        session.commit.assert_not_awaited()

    @pytest.mark.parametrize(
        "status",
        [JobStatus.RUNNING.value, JobStatus.PENDING.value],
    )
    async def test_publishes_actual_previous_status_and_only_public_message(
        self,
        status: str,
    ) -> None:
        job = _make_job(status)
        job.error_message = "internal-host.example.test sentinel-secret"
        job.public_error_message = None
        bus = AsyncMock()
        svc, session, _ = _make_service(job, event_bus=bus)

        async def refresh_job(refreshed: MagicMock) -> None:
            refreshed.status = JobStatus.FAILED.value
            refreshed.public_error_message = "The AI provider is temporarily unavailable."
            refreshed.failure_code = "provider_unavailable"

        session.refresh.side_effect = refresh_job
        await svc.transition_to_failed(
            job.id,
            error_message=job.error_message,
            public_error_message="The AI provider is temporarily unavailable.",
            failure_code="provider_unavailable",
            refund=False,
            product_id="vex",
        )

        payload = bus.publish.await_args.kwargs["payload"]
        assert payload.previous_status == status
        assert payload.error_message == "The AI provider is temporarily unavailable."
        assert "sentinel-secret" not in payload.error_message


# ---------------------------------------------------------------------------
# make_output_expires_at
# ---------------------------------------------------------------------------


class TestMakeOutputExpiresAt:
    def test_returns_future_datetime(self) -> None:
        result = JobStateTransitionService.make_output_expires_at(7)
        assert result > datetime.now(UTC)

    def test_respects_retention_days(self) -> None:
        now = datetime.now(UTC)
        result = JobStateTransitionService.make_output_expires_at(14)
        assert result > now + timedelta(days=13)
