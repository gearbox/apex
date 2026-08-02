"""Unit tests for JobSweepService."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.services.generation.aisha_failures import AishaFailure
from src.api.services.jobs.sweep import JobSweepFailure, JobSweepResult, JobSweepService
from src.core.enums import JobStatus
from src.db.models.storage import GenerationJob

_REPO_PATH = "src.api.services.jobs.sweep.JobRepository"
_TRANSITION_PATH = "src.api.services.jobs.sweep.JobStateTransitionService"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def _make_job(*, status: str = JobStatus.QUEUED.value, **kwargs: Any) -> MagicMock:
    job = MagicMock(spec=GenerationJob)
    job.id = uuid4()
    job.status = status
    job.error_message = None
    for k, v in kwargs.items():
        setattr(job, k, v)
    return job


def _make_mock_session_factory() -> tuple[MagicMock, MagicMock]:
    """Return (factory, db_session) mocks supporting 'async with factory() as db'."""
    mock_db = MagicMock()
    mock_db.__aenter__ = AsyncMock(return_value=mock_db)
    mock_db.__aexit__ = AsyncMock(return_value=None)
    mock_factory = MagicMock(return_value=mock_db)
    return mock_factory, mock_db


def _make_service(**overrides: Any) -> tuple[JobSweepService, dict[str, Any]]:
    mock_factory, mock_db = _make_mock_session_factory()
    billing = AsyncMock()
    mocks: dict[str, Any] = {
        "session_factory": mock_factory,
        "mock_db": mock_db,
        "event_bus": None,
        "billing_service": billing,
    } | overrides
    svc = JobSweepService(
        session_factory=mocks["session_factory"],
        event_bus=mocks["event_bus"],
        billing_service=mocks["billing_service"],
    )
    return svc, mocks


# ---------------------------------------------------------------------------
# Tests: sweep_session
# ---------------------------------------------------------------------------


class TestSweepSession:
    async def test_sweep_no_jobs_returns_zero_counts(self) -> None:
        svc, _ = _make_service()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = []

            result = await svc.sweep_session(
                uuid4(), product_id="vex", failure=JobSweepFailure("session stopped")
            )

        assert result == JobSweepResult(swept_count=0, error_count=0, skipped_count=0)

    async def test_sweep_two_jobs_both_transitioned_swept_count_2(self) -> None:
        svc, _ = _make_service()
        job1 = _make_job()
        job2 = _make_job()

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job1, job2]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.side_effect = [
                (job1, True),
                (job2, True),
            ]

            result = await svc.sweep_session(
                uuid4(),
                product_id="vex",
                failure=JobSweepFailure("GPU session stopped before job completed."),
            )

        assert result.swept_count == 2
        assert result.error_count == 0
        assert result.skipped_count == 0

    async def test_already_terminal_job_counted_as_skipped(self) -> None:
        """transition_to_failed returning did=False → counted as skipped."""
        svc, _ = _make_service()
        job = _make_job()

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = (job, False)

            result = await svc.sweep_session(
                uuid4(),
                product_id="vex",
                failure=JobSweepFailure("GPU session stopped before job completed."),
            )

        assert result.swept_count == 0
        assert result.skipped_count == 1
        assert result.error_count == 0

    async def test_mixed_did_true_and_false_counted_correctly(self) -> None:
        svc, _ = _make_service()
        job1 = _make_job()
        job2 = _make_job()

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job1, job2]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.side_effect = [
                (job1, True),
                (job2, False),
            ]

            result = await svc.sweep_session(
                uuid4(), product_id="vex", failure=JobSweepFailure("stopped")
            )

        assert result.swept_count == 1
        assert result.skipped_count == 1
        assert result.error_count == 0

    async def test_sweep_one_job_errors_other_succeeds_isolated(self) -> None:
        """Per-job exception is isolated; the other job still transitions."""
        svc, _ = _make_service()
        job1 = _make_job()
        job2 = _make_job()

        call_count = 0

        async def side_effect(*_args: object, **_kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient DB error")
            return (job2, True)

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job1, job2]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.side_effect = side_effect

            result = await svc.sweep_session(
                uuid4(),
                product_id="vex",
                failure=JobSweepFailure("GPU session stopped before job completed."),
            )

        assert result.error_count == 1
        assert result.swept_count == 1

    async def test_sweep_uses_single_session_for_all_jobs(self) -> None:
        """session_factory() is called exactly once for the entire sweep."""
        svc, mocks = _make_service()
        jobs = [_make_job(), _make_job()]

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = jobs

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.side_effect = [(j, True) for j in jobs]

            await svc.sweep_session(uuid4(), product_id="vex", failure=JobSweepFailure("stopped"))

        # Single session for the whole sweep
        assert mocks["session_factory"].call_count == 1

    async def test_sweep_passes_correct_product_id_to_transition_to_failed(self) -> None:
        svc, _ = _make_service()
        job = _make_job()

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = (job, True)

            await svc.sweep_session(
                uuid4(), product_id="synthara", failure=JobSweepFailure("stopped")
            )

            mock_ts.transition_to_failed.assert_awaited_once()
            call_kwargs = mock_ts.transition_to_failed.call_args.kwargs
            assert call_kwargs["product_id"] == "synthara"
            assert call_kwargs["failure_code"] == AishaFailure.GENERATION_SESSION_TERMINATED.value
            assert (
                call_kwargs["public_error_message"]
                == AishaFailure.GENERATION_SESSION_TERMINATED.public_message
            )

    async def test_sweep_truncates_reason_to_500_chars(self) -> None:
        svc, _ = _make_service()
        job = _make_job()
        long_reason = "x" * 600

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = (job, True)

            await svc.sweep_session(uuid4(), product_id="vex", failure=JobSweepFailure(long_reason))

            call_kwargs = mock_ts.transition_to_failed.call_args.kwargs
            assert len(call_kwargs["error_message"]) == 500

    async def test_sweep_refund_is_true(self) -> None:
        """Sweep always passes refund=True to transition_to_failed."""
        svc, _ = _make_service()
        job = _make_job()

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = (job, True)

            await svc.sweep_session(uuid4(), product_id="vex", failure=JobSweepFailure("stopped"))

            call_kwargs = mock_ts.transition_to_failed.call_args.kwargs
            assert call_kwargs["refund"] is True


# ---------------------------------------------------------------------------
# Tests: sweep_session_best_effort
# ---------------------------------------------------------------------------


class TestSweepSessionBestEffort:
    async def test_logs_on_success(self) -> None:
        svc, _ = _make_service()
        session_id = uuid4()

        with (
            patch.object(
                svc,
                "sweep_session",
                new_callable=AsyncMock,
                return_value=JobSweepResult(swept_count=2, error_count=0, skipped_count=1),
            ),
            patch("src.api.services.jobs.sweep.logger") as mock_logger,
        ):
            await svc.sweep_session_best_effort(
                session_id=session_id,
                product_id="vex",
                failure=JobSweepFailure("stopped"),
                log_event="gpu_session.stop.job_sweep",
            )

        mock_logger.info.assert_called_once_with(
            "gpu_session.stop.job_sweep",
            session_id=str(session_id),
            swept_count=2,
            error_count=0,
            skipped_count=1,
        )

    async def test_swallows_exception_and_logs(self) -> None:
        svc, _ = _make_service()

        with patch.object(
            svc,
            "sweep_session",
            new_callable=AsyncMock,
            side_effect=RuntimeError("DB exploded"),
        ):
            # Must not raise
            await svc.sweep_session_best_effort(
                session_id=uuid4(),
                product_id="vex",
                failure=JobSweepFailure("stopped"),
                log_event="gpu_session.stop.job_sweep",
            )

    async def test_passes_args_through_to_sweep_session(self) -> None:
        svc, _ = _make_service()
        session_id = uuid4()

        with patch.object(
            svc,
            "sweep_session",
            new_callable=AsyncMock,
            return_value=JobSweepResult(swept_count=0, error_count=0, skipped_count=0),
        ) as mock_sweep:
            await svc.sweep_session_best_effort(
                session_id=session_id,
                product_id="synthara",
                failure=JobSweepFailure("provisioning failed"),
                log_event="gpu_session.provision.job_sweep",
            )

        mock_sweep.assert_awaited_once_with(
            session_id,
            product_id="synthara",
            failure=JobSweepFailure("provisioning failed"),
        )
