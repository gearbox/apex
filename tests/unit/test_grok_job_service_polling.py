"""Tests for GrokJobService video-poll error taxonomy (H1 remediation).

Rate-limit and timeout errors from xAI are transient and must not fail a
healthy in-flight job. ``_poll_video_result`` re-raises them; each entry
point reacts differently:
- ``poll_video_job_for_worker`` (worker path): propagates, so the worker can
  skip the tick without transitioning or refunding.
- ``poll_video_job`` (read-through path): catches them and reports the job
  unchanged, so a status request never 500s on an xAI hiccup.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.grok import GrokModerationError, GrokRateLimitError, GrokTimeoutError
from src.api.services.grok.job_service import GrokJobService, VideoPollOutcome
from src.core.enums import JobStatus, VideoPollStatus


def _make_service() -> GrokJobService:
    return GrokJobService(grok_client=AsyncMock(), storage=MagicMock())


def _make_job(*, status: str = JobStatus.RUNNING.value) -> MagicMock:
    job = MagicMock()
    job.id = uuid4()
    job.user_id = uuid4()
    job.product_id = "vex"
    job.status = status
    job.external_request_id = "grok-req-1"
    return job


def _patched_job_repository(job: object) -> tuple[Any, AsyncMock]:
    mock_repo = MagicMock()
    mock_repo.get = AsyncMock(return_value=job)
    mock_repo.update_status = AsyncMock(side_effect=lambda job_id, status, **kw: job)  # noqa: ARG005
    patcher = patch(
        "src.api.services.grok.job_service.JobRepository",
        return_value=mock_repo,
    )
    return patcher, mock_repo


class TestPollVideoJobForWorkerTransientPropagation:
    async def test_poll_video_result_reraises_rate_limit(self) -> None:
        job = _make_job()
        service = _make_service()
        service._grok.get_video_result = AsyncMock(side_effect=GrokRateLimitError("rate limited"))
        patcher, _ = _patched_job_repository(job)

        with patcher, pytest.raises(GrokRateLimitError):
            await service.poll_video_job_for_worker(AsyncMock(), job.id)

    async def test_poll_video_result_reraises_timeout(self) -> None:
        job = _make_job()
        service = _make_service()
        service._grok.get_video_result = AsyncMock(
            side_effect=GrokTimeoutError("deadline exceeded")
        )
        patcher, _ = _patched_job_repository(job)

        with patcher, pytest.raises(GrokTimeoutError):
            await service.poll_video_job_for_worker(AsyncMock(), job.id)

    async def test_poll_video_result_moderation_error_is_failed_outcome(self) -> None:
        job = _make_job()
        service = _make_service()
        service._grok.get_video_result = AsyncMock(
            side_effect=GrokModerationError("content rejected")
        )
        patcher, _ = _patched_job_repository(job)

        with patcher:
            outcome = await service.poll_video_job_for_worker(AsyncMock(), job.id)

        assert outcome == VideoPollOutcome(
            status=VideoPollStatus.FAILED, error_message="content rejected"
        )


class TestReadThroughTransientMapping:
    async def test_read_through_maps_transient_to_still_running(self) -> None:
        job = _make_job(status=JobStatus.RUNNING.value)
        service = _make_service()
        service._grok.get_video_result = AsyncMock(side_effect=GrokRateLimitError("rate limited"))
        patcher, mock_repo = _patched_job_repository(job)

        with patcher:
            result = await service.poll_video_job(AsyncMock(), job.id)

        assert result is job
        mock_repo.update_status.assert_not_awaited()
