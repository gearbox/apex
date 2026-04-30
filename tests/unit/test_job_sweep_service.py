"""Unit tests for JobSweepService."""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.jobs.sweep import JobSweepResult, JobSweepService
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
# Tests
# ---------------------------------------------------------------------------


class TestSweepSession:
    async def test_sweep_no_jobs_returns_zero_counts(self) -> None:
        svc, _ = _make_service()

        with patch(_REPO_PATH) as MockRepo:
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = []

            result = await svc.sweep_session(uuid4(), product_id="vex", reason="session stopped")

        assert result == JobSweepResult(swept_count=0, error_count=0, skipped_count=0)

    async def test_sweep_two_jobs_both_transitioned_swept_count_2(self) -> None:
        svc, _ = _make_service()
        reason = "GPU session stopped before job completed."
        job1 = _make_job()
        job2 = _make_job()
        failed_row1 = MagicMock()
        failed_row1.status = JobStatus.FAILED.value
        failed_row1.error_message = reason[:500]
        failed_row2 = MagicMock()
        failed_row2.status = JobStatus.FAILED.value
        failed_row2.error_message = reason[:500]

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job1, job2]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.side_effect = [failed_row1, failed_row2]

            result = await svc.sweep_session(uuid4(), product_id="vex", reason=reason)

        assert result.swept_count == 2
        assert result.error_count == 0
        assert result.skipped_count == 0

    async def test_sweep_already_failed_job_skipped_count_1(self) -> None:
        """Job already FAILED with a different message → counted as skipped."""
        svc, _ = _make_service()
        job = _make_job()
        already_failed = MagicMock()
        already_failed.status = JobStatus.FAILED.value
        already_failed.error_message = "some other reason"

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = already_failed

            result = await svc.sweep_session(
                uuid4(), product_id="vex", reason="GPU session stopped before job completed."
            )

        assert result.swept_count == 0
        assert result.skipped_count == 1
        assert result.error_count == 0

    async def test_sweep_one_job_errors_other_succeeds_isolated(self) -> None:
        """Per-job exception is isolated; the other job still transitions."""
        svc, _ = _make_service()
        reason = "GPU session stopped before job completed."
        job1 = _make_job()
        job2 = _make_job()
        failed_row = MagicMock()
        failed_row.status = JobStatus.FAILED.value
        failed_row.error_message = reason[:500]

        call_count = 0

        async def side_effect(job_id: object, **kwargs: object) -> object:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("transient DB error")
            return failed_row

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

            result = await svc.sweep_session(uuid4(), product_id="vex", reason=reason)

        assert result.error_count == 1
        assert result.swept_count == 1

    async def test_sweep_each_job_uses_own_session(self) -> None:
        """session_factory() is called once for the snapshot + once per job."""
        svc, mocks = _make_service()
        jobs = [_make_job(), _make_job()]
        reason = "stopped"
        failed_row = MagicMock()
        failed_row.status = JobStatus.FAILED.value
        failed_row.error_message = reason[:500]

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = jobs

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = failed_row

            await svc.sweep_session(uuid4(), product_id="vex", reason=reason)

        # 1 call for snapshot + 1 call per job = 3 total
        assert mocks["session_factory"].call_count == 1 + len(jobs)

    async def test_sweep_passes_correct_product_id_to_transition_to_failed(self) -> None:
        svc, _ = _make_service()
        job = _make_job()
        reason = "GPU session stopped before job completed."
        failed_row = MagicMock()
        failed_row.status = JobStatus.FAILED.value
        failed_row.error_message = reason[:500]

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = failed_row

            await svc.sweep_session(uuid4(), product_id="synthara", reason=reason)

            mock_ts.transition_to_failed.assert_awaited_once()
            call_kwargs = mock_ts.transition_to_failed.call_args.kwargs
            assert call_kwargs["product_id"] == "synthara"

    async def test_sweep_truncates_reason_to_500_chars(self) -> None:
        svc, _ = _make_service()
        job = _make_job()
        long_reason = "x" * 600
        failed_row = MagicMock()
        failed_row.status = JobStatus.FAILED.value
        failed_row.error_message = ("x" * 600)[:500]

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = failed_row

            await svc.sweep_session(uuid4(), product_id="vex", reason=long_reason)

            call_kwargs = mock_ts.transition_to_failed.call_args.kwargs
            assert len(call_kwargs["error_message"]) == 500

    async def test_sweep_refund_is_true(self) -> None:
        """Sweep always passes refund=True to transition_to_failed."""
        svc, _ = _make_service()
        job = _make_job()
        reason = "stopped"
        failed_row = MagicMock()
        failed_row.status = JobStatus.FAILED.value
        failed_row.error_message = reason[:500]

        with (
            patch(_REPO_PATH) as MockRepo,
            patch(_TRANSITION_PATH) as MockTS,
        ):
            mock_repo = AsyncMock()
            MockRepo.return_value = mock_repo
            mock_repo.list_in_flight_for_session.return_value = [job]

            mock_ts = AsyncMock()
            MockTS.return_value = mock_ts
            mock_ts.transition_to_failed.return_value = failed_row

            await svc.sweep_session(uuid4(), product_id="vex", reason=reason)

            call_kwargs = mock_ts.transition_to_failed.call_args.kwargs
            assert call_kwargs["refund"] is True
