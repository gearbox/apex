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

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.api.services.grok import (
    GrokDeferredTerminalError,
    GrokModerationError,
    GrokRateLimitError,
    GrokTimeoutError,
)
from src.api.services.grok.job_service import GrokJobService
from src.core.enums import JobStatus, Provider, VideoPollStatus


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


def test_video_job_with_missing_started_at_uses_created_at_for_ttl() -> None:
    service = GrokJobService(grok_client=AsyncMock(), storage=MagicMock(), max_poll_time=60)
    job = _make_job()
    job.started_at = None
    job.created_at = datetime.now(UTC) - timedelta(seconds=61)

    assert service._is_video_job_overdue(job) is True


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

        assert outcome.status == VideoPollStatus.FAILED
        assert outcome.error_message == (
            "The requested content was rejected by the AI provider's safety system. "
            "This generation was charged because the provider processed the request. "
            "Modify the prompt or input and try again."
        )
        assert outcome.failure is not None
        assert outcome.failure.kind == ProviderFailureKind.MODERATION_REJECTED
        assert outcome.failure.billable is True

    @pytest.mark.parametrize(
        "kind",
        [ProviderFailureKind.RATE_LIMITED, ProviderFailureKind.TIMEOUT],
    )
    async def test_terminal_deferred_retryable_kind_is_failed_outcome(
        self,
        kind: ProviderFailureKind,
    ) -> None:
        job = _make_job()
        service = _make_service()
        failure = ProviderFailure(
            kind=kind,
            provider=Provider.GROK,
            sanitized_message=ProviderFailure.safe_message_for_kind(kind),
            retryable=True,
            provider_request_accepted=True,
        )
        service._grok.get_video_result = AsyncMock(
            side_effect=GrokDeferredTerminalError("provider marked it failed", failure=failure)
        )
        patcher, _ = _patched_job_repository(job)

        with patcher:
            outcome = await service.poll_video_job_for_worker(AsyncMock(), job.id)

        assert outcome.status == VideoPollStatus.FAILED
        assert outcome.failure is not None
        assert outcome.failure.kind == kind


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

    async def test_read_through_terminal_failure_uses_shared_settlement(self) -> None:
        job = _make_job(status=JobStatus.QUEUED.value)
        service = _make_service()
        failure = ProviderFailure(
            kind=ProviderFailureKind.PROVIDER_UNAVAILABLE,
            provider=Provider.GROK,
            sanitized_message="The AI provider is temporarily unavailable.",
        )
        outcome = MagicMock(status=VideoPollStatus.FAILED, failure=failure)
        service._poll_video_result = AsyncMock(return_value=outcome)  # type: ignore[method-assign]
        service.settle_video_poll_outcome = AsyncMock(return_value=job)  # type: ignore[method-assign]
        patcher, _ = _patched_job_repository(job)
        session = AsyncMock()

        with patcher:
            result = await service.poll_video_job(session, job.id)

        assert result is job
        service.settle_video_poll_outcome.assert_awaited_once_with(
            session,
            job_id=job.id,
            outcome=outcome,
            product_id="vex",
        )
