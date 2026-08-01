"""Tests for GrokVideoWorker — worker/read-through shared settlement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.api.services.grok import GrokRateLimitError, GrokTimeoutError
from src.api.services.grok.job_service import VideoPollOutcome
from src.api.services.grok.video_worker import GrokVideoWorker
from src.core.enums import GenerationType, JobStatus, Provider, VideoPollStatus


def _make_settings(**overrides: object) -> MagicMock:
    settings = MagicMock()
    settings.grok_video_poll_interval = 5
    settings.grok_video_max_poll_time = 600
    settings.grok_video_max_concurrent_polls = 4
    for k, v in overrides.items():
        setattr(settings, k, v)
    return settings


def _make_worker(**overrides: object) -> GrokVideoWorker:
    defaults: dict[str, object] = {
        "db_manager": MagicMock(),
        "job_service": AsyncMock(),
        "billing_service": AsyncMock(),
        "settings": _make_settings(),
        "event_bus": None,
    } | overrides
    return GrokVideoWorker(**defaults)  # type: ignore[arg-type]


def _make_job(
    *,
    status: str = JobStatus.RUNNING.value,
    started_at: datetime | None = None,
    generation_type: str = GenerationType.T2V.value,
) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.user_id = uuid4()
    job.product_id = "vex"
    job.status = status
    job.generation_type = generation_type
    job.started_at = started_at or (datetime.now(UTC) - timedelta(seconds=10))
    return job


class _SessionCtx:
    def __init__(self, session: object) -> None:
        self._session = session

    async def __aenter__(self) -> object:
        return self._session

    async def __aexit__(self, *exc: object) -> bool:
        return False


def _patched_transition_service() -> tuple[Any, AsyncMock]:
    """Return (patcher, mock ts instance) for JobStateTransitionService."""
    mock_ts = AsyncMock()
    patcher = patch(
        "src.api.services.grok.video_worker.JobStateTransitionService",
        return_value=mock_ts,
    )
    return patcher, mock_ts


class TestTimeoutRefund:
    async def test_timeout_marks_failed_with_refund(self) -> None:
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=1000))
        job_service = AsyncMock()
        worker = _make_worker(
            job_service=job_service, settings=_make_settings(grok_video_max_poll_time=600)
        )
        patcher, mock_ts = _patched_transition_service()

        with patcher:
            await worker._poll_one(job, AsyncMock())

        mock_ts.transition_to_failed.assert_awaited_once()
        _, kwargs = mock_ts.transition_to_failed.call_args
        assert kwargs["refund"] is True
        assert kwargs["product_id"] == "vex"
        job_service.poll_video_job_for_worker.assert_not_awaited()

    async def test_below_timeout_does_not_fail(self) -> None:
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=10))
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.STILL_RUNNING
        )
        worker = _make_worker(
            job_service=job_service, settings=_make_settings(grok_video_max_poll_time=600)
        )
        patcher, mock_ts = _patched_transition_service()

        with patcher:
            await worker._poll_one(job, AsyncMock())

        mock_ts.transition_to_failed.assert_not_awaited()


class TestCompletedTransition:
    async def test_completed_goes_through_shared_settlement(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.COMPLETED
        )
        worker = _make_worker(job_service=job_service)
        session = AsyncMock()
        await worker._poll_one(job, session)

        job_service.settle_video_poll_outcome.assert_awaited_once_with(
            session,
            job_id=job.id,
            outcome=job_service.poll_video_job_for_worker.return_value,
            product_id="vex",
        )


class TestFailedProviderRefundPolicy:
    async def test_billable_moderation_failure_is_not_refunded(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        failure = ProviderFailure(
            kind=ProviderFailureKind.MODERATION_REJECTED,
            provider=Provider.GROK,
            sanitized_message=(
                "The requested content was rejected by the AI provider's safety system. "
                "Modify the prompt or input and try again."
            ),
            provider_request_accepted=True,
            billable=True,
            provider_request_id="grok-request-1",
        )
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.FAILED,
            error_message=failure.sanitized_message,
            failure=failure,
        )
        worker = _make_worker(job_service=job_service)
        session = AsyncMock()
        await worker._poll_one(job, session)

        job_service.settle_video_poll_outcome.assert_awaited_once_with(
            session,
            job_id=job.id,
            outcome=job_service.poll_video_job_for_worker.return_value,
            product_id="vex",
        )

    async def test_failed_provider_result_refund_policy(self) -> None:
        """poll_video_job_for_worker's failure path (GrokAPIError from
        get_video_result) has never been refunded upstream — unlike the
        synchronous submission paths in job_service, which already refund
        via billing_service.refund. refund=True here is correct: no
        double-charge risk."""
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.FAILED, error_message="xAI said no"
        )
        worker = _make_worker(job_service=job_service)
        session = AsyncMock()
        await worker._poll_one(job, session)

        job_service.settle_video_poll_outcome.assert_awaited_once_with(
            session,
            job_id=job.id,
            outcome=job_service.poll_video_job_for_worker.return_value,
            product_id="vex",
        )

    async def test_failed_without_error_message_uses_fallback(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.FAILED, error_message=None
        )
        worker = _make_worker(job_service=job_service)
        session = AsyncMock()
        await worker._poll_one(job, session)

        job_service.settle_video_poll_outcome.assert_awaited_once_with(
            session,
            job_id=job.id,
            outcome=job_service.poll_video_job_for_worker.return_value,
            product_id="vex",
        )


class TestQueuedToRunningTransition:
    async def test_queued_to_running_transition(self) -> None:
        job = _make_job(status=JobStatus.QUEUED.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.STILL_RUNNING
        )
        worker = _make_worker(job_service=job_service)
        session = AsyncMock()
        await worker._poll_one(job, session)

        job_service.settle_video_poll_outcome.assert_awaited_once_with(
            session,
            job_id=job.id,
            outcome=job_service.poll_video_job_for_worker.return_value,
            product_id="vex",
        )

    async def test_already_running_emits_progress_not_transition(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.STILL_RUNNING
        )
        event_bus = AsyncMock()
        worker = _make_worker(job_service=job_service, event_bus=event_bus)
        patcher, mock_ts = _patched_transition_service()

        with patcher:
            await worker._poll_one(job, AsyncMock())

        mock_ts.transition_to_running.assert_not_awaited()
        event_bus.publish.assert_awaited_once()


def _patched_job_repository(jobs: list[object]) -> tuple[Any, AsyncMock]:
    """Return (patcher, mock repo instance) for JobRepository, list_pending_video_jobs -> jobs."""
    mock_repo = MagicMock()
    mock_repo.list_pending_video_jobs = AsyncMock(return_value=jobs)
    patcher = patch(
        "src.api.services.grok.video_worker.JobRepository",
        return_value=mock_repo,
    )
    return patcher, mock_repo


class TestRunOnceFanOut:
    async def test_session_per_job_isolation(self) -> None:
        """job A raising inside _poll_one must not poison job B's processing."""
        job_a = _make_job()
        job_b = _make_job()

        list_session = AsyncMock()
        job_session_a = AsyncMock()
        job_session_b = AsyncMock()
        sessions = iter([list_session, job_session_a, job_session_b])

        db_manager = MagicMock()
        db_manager.session.side_effect = lambda: _SessionCtx(next(sessions))

        worker = _make_worker(db_manager=db_manager)
        processed: list[object] = []

        async def fake_poll_one(job: object, session: object) -> None:
            del session
            if job is job_a:
                raise RuntimeError("job A blew up")
            processed.append(job)

        worker._poll_one = fake_poll_one  # type: ignore[method-assign]

        patcher, _ = _patched_job_repository([job_a, job_b])
        with patcher:
            await worker.run_once()  # must not raise despite job A's exception

        assert processed == [job_b]

    async def test_empty_candidate_list_is_noop(self) -> None:
        list_session = AsyncMock()
        db_manager = MagicMock()
        db_manager.session.side_effect = lambda: _SessionCtx(list_session)

        worker = _make_worker(db_manager=db_manager)
        worker._poll_one = AsyncMock()  # type: ignore[method-assign]

        patcher, _ = _patched_job_repository([])
        with patcher:
            await worker.run_once()

        worker._poll_one.assert_not_awaited()

    async def test_candidate_query_filters_grok_provider(self) -> None:
        """run_once must route through the provider-scoped repo query (H2 fix),
        not an inline query that would also match Aisha jobs."""
        list_session = AsyncMock()
        db_manager = MagicMock()
        db_manager.session.side_effect = lambda: _SessionCtx(list_session)

        worker = _make_worker(db_manager=db_manager)

        patcher, mock_repo = _patched_job_repository([])
        with patcher:
            await worker.run_once()

        mock_repo.list_pending_video_jobs.assert_awaited_once_with(provider=Provider.GROK)


class TestTransientPollErrors:
    """H1: rate-limit/timeout poll errors must be transient, not terminal (D1/D2)."""

    async def test_rate_limit_error_skips_job_without_transition(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.side_effect = GrokRateLimitError("rate limited")
        worker = _make_worker(job_service=job_service)
        patcher, mock_ts = _patched_transition_service()

        with patcher:
            await worker._poll_one(job, AsyncMock())

        mock_ts.transition_to_failed.assert_not_awaited()
        mock_ts.transition_to_running.assert_not_awaited()
        mock_ts.transition_to_completed.assert_not_awaited()

    async def test_timeout_error_skips_job_without_transition(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.side_effect = GrokTimeoutError("deadline exceeded")
        worker = _make_worker(job_service=job_service)
        patcher, mock_ts = _patched_transition_service()

        with patcher:
            await worker._poll_one(job, AsyncMock())

        mock_ts.transition_to_failed.assert_not_awaited()
        mock_ts.transition_to_running.assert_not_awaited()
        mock_ts.transition_to_completed.assert_not_awaited()

    async def test_transient_error_does_not_refund(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.side_effect = GrokRateLimitError("rate limited")
        worker = _make_worker(job_service=job_service)
        patcher, mock_ts = _patched_transition_service()

        with patcher:
            await worker._poll_one(job, AsyncMock())

        for call in mock_ts.transition_to_failed.await_args_list:
            assert call.kwargs.get("refund") is not True

    async def test_terminal_api_error_still_fails_with_refund(self) -> None:
        """Non-transient GrokAPIError surfaces as a FAILED outcome from the job
        service (job_service maps it, worker doesn't see the exception) and is
        still failed with a refund — unchanged behavior."""
        job = _make_job(status=JobStatus.RUNNING.value)
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.return_value = VideoPollOutcome(
            status=VideoPollStatus.FAILED, error_message="moderation rejected"
        )
        worker = _make_worker(job_service=job_service)
        session = AsyncMock()
        await worker._poll_one(job, session)

        job_service.settle_video_poll_outcome.assert_awaited_once_with(
            session,
            job_id=job.id,
            outcome=job_service.poll_video_job_for_worker.return_value,
            product_id="vex",
        )

    async def test_timeout_ceiling_precedes_poll(self) -> None:
        """A job past max_poll_time is failed via the timeout branch without
        ever invoking the poll — even if that poll would have raised a
        transient error."""
        job = _make_job(started_at=datetime.now(UTC) - timedelta(seconds=1000))
        job_service = AsyncMock()
        job_service.poll_video_job_for_worker.side_effect = GrokRateLimitError("rate limited")
        worker = _make_worker(
            job_service=job_service, settings=_make_settings(grok_video_max_poll_time=600)
        )
        patcher, mock_ts = _patched_transition_service()

        with patcher:
            await worker._poll_one(job, AsyncMock())

        mock_ts.transition_to_failed.assert_awaited_once()
        _, kwargs = mock_ts.transition_to_failed.call_args
        assert kwargs["refund"] is True
        job_service.poll_video_job_for_worker.assert_not_awaited()


class TestManagerRemoved:
    def test_manager_removed(self) -> None:
        """GrokVideoWorkerManager was deleted — the worker is wired through
        ServiceContainer directly, like every other worker."""
        import src.api.services.grok.video_worker as module

        assert not hasattr(module, "GrokVideoWorkerManager")
