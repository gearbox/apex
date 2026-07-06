"""Tests for the per-provider capability protocol (F7/D5).

Covers:
- ``supports_image_sizing`` — GenerationService skips request-level image
  sizing fields for any provider that declares it doesn't honour them
  (previously a hardcoded ``== Provider.GROK`` check).
- ``refresh_job`` — the poll-on-read hook. Default no-op (Aisha, and the
  Protocol's own declared default); Grok's real implementation gates on
  video generation type and non-terminal status (moved out of
  ``UnifiedJobService``, which now just delegates unconditionally — see
  ``tests/unit/test_jobs.py`` for its dispatch-by-provider-key coverage).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import ANY, AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.generation.aisha_provider import AishaGenerationProvider
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.grok_provider import GrokGenerationProvider
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import GenerationService
from src.api.services.unified_jobs import UnifiedJobService
from src.core.enums import GenerationType, JobStatus, ModelType, Provider, Resolution
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


def _make_enabled_model_mock() -> MagicMock:
    record = MagicMock()
    record.is_enabled = True
    return record


class TestSupportsImageSizingFlag:
    async def test_sizing_ignored_for_provider_without_image_sizing(self) -> None:
        """A provider declaring supports_image_sizing=False has its
        image_resolution/width/height silently ignored (logged, not applied)
        — the check is flag-driven, not a hardcoded Provider.GROK comparison."""
        job = MagicMock()
        job.id = uuid4()
        job.status = JobStatus.COMPLETED.value
        job.name = "Test"
        job.created_at = datetime.now(UTC)

        provider = MagicMock()
        provider.supports_image_sizing = False
        provider.validate = MagicMock()
        provider.submit = AsyncMock(
            return_value=ProviderSubmitResult(job=job, balance_after=100, balance_event=None)
        )

        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing.assert_sufficient_balance = AsyncMock()
        billing.get_balance = AsyncMock(return_value=1000)

        pricing = AsyncMock()
        pricing.get_price = AsyncMock(return_value=50)

        service = GenerationService(
            providers={Provider.GROK: provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )

        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            image_resolution=Resolution.ULTRA,
            width=None,
            height=None,
        )

        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.api.services.generation.service.logger") as mock_logger,
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        mock_logger.info.assert_any_call(
            "generation.image_sizing.ignored_for_grok",
            model=request.model.value,
            image_resolution="ultra",
            width=None,
            height=None,
        )

    async def test_sizing_honoured_for_provider_with_image_sizing(self) -> None:
        """The default flag (True) — image sizing fields are NOT reported as
        ignored for a provider that honours them."""
        job = MagicMock()
        job.id = uuid4()
        job.status = JobStatus.COMPLETED.value
        job.name = "Test"
        job.created_at = datetime.now(UTC)

        provider = MagicMock()
        provider.supports_image_sizing = True
        provider.validate = MagicMock()
        provider.submit = AsyncMock(
            return_value=ProviderSubmitResult(job=job, balance_after=100, balance_event=None)
        )

        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing.assert_sufficient_balance = AsyncMock()
        billing.get_balance = AsyncMock(return_value=1000)

        pricing = AsyncMock()
        pricing.get_price = AsyncMock(return_value=50)

        service = GenerationService(
            providers={Provider.AISHA: provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )

        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            image_resolution=Resolution.ULTRA,
        )

        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.api.services.generation.service.logger") as mock_logger,
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        assert ("generation.image_sizing.ignored_for_grok",) not in [
            call.args for call in mock_logger.info.call_args_list
        ]


class TestRefreshJobDefaultNoop:
    async def test_refresh_job_default_noop(self) -> None:
        """AishaGenerationProvider has no async backend to poll — refresh_job
        is an unconditional no-op, regardless of job state."""
        provider = AishaGenerationProvider(
            workflow_service=MagicMock(),
            gpu_session_service=MagicMock(),
            bundle_index=MagicMock(),
        )
        job = MagicMock()
        job.status = JobStatus.QUEUED.value
        job.generation_type = GenerationType.T2I.value

        result = await provider.refresh_job(AsyncMock(), job)

        assert result is None


class TestGrokRefreshJobGating:
    """GrokGenerationProvider.refresh_job's gating logic (moved out of
    UnifiedJobService in D5): only polls video jobs that are still
    queued/running."""

    def _make_provider(
        self, *, poll_return: object = None
    ) -> tuple[GrokGenerationProvider, AsyncMock]:
        grok_job_service = AsyncMock()
        grok_job_service.poll_video_job = AsyncMock(return_value=poll_return)
        provider = GrokGenerationProvider(grok_job_service=grok_job_service, r2_storage=AsyncMock())
        return provider, grok_job_service

    def _make_job(self, *, generation_type: str, status: str) -> MagicMock:
        job = MagicMock()
        job.id = uuid4()
        job.generation_type = generation_type
        job.status = status
        return job

    async def test_polls_queued_video_job(self) -> None:
        updated = MagicMock()
        provider, grok_job_service = self._make_provider(poll_return=updated)
        job = self._make_job(
            generation_type=GenerationType.T2V.value, status=JobStatus.QUEUED.value
        )

        result = await provider.refresh_job(AsyncMock(), job)

        grok_job_service.poll_video_job.assert_awaited_once_with(ANY, job.id)
        assert result is updated

    async def test_does_not_poll_image_job(self) -> None:
        provider, grok_job_service = self._make_provider()
        job = self._make_job(
            generation_type=GenerationType.T2I.value, status=JobStatus.RUNNING.value
        )

        result = await provider.refresh_job(AsyncMock(), job)

        grok_job_service.poll_video_job.assert_not_awaited()
        assert result is None

    async def test_does_not_poll_completed_video_job(self) -> None:
        provider, grok_job_service = self._make_provider()
        job = self._make_job(
            generation_type=GenerationType.T2V.value, status=JobStatus.COMPLETED.value
        )

        result = await provider.refresh_job(AsyncMock(), job)

        grok_job_service.poll_video_job.assert_not_awaited()
        assert result is None

    async def test_returns_updated_job_on_successful_poll(self) -> None:
        updated = MagicMock()
        provider, _grok_job_service = self._make_provider(poll_return=updated)
        job = self._make_job(
            generation_type=GenerationType.I2V.value, status=JobStatus.RUNNING.value
        )

        result = await provider.refresh_job(AsyncMock(), job)

        assert result is updated


class TestUnifiedJobsDelegation:
    async def test_unified_jobs_refresh_delegates_to_provider(self) -> None:
        """UnifiedJobService resolves job.provider to a registered provider
        and calls refresh_job(session, job) unconditionally — no gating logic
        of its own (that lives inside each provider now)."""
        from tests.unit.test_jobs import _make_job, _session_for_get

        user_id = uuid4()
        job = _make_job(
            user_id=user_id,
            provider="grok",
            generation_type=GenerationType.T2V.value,
            status=JobStatus.QUEUED.value,
        )
        session = _session_for_get(job)

        grok_provider = AsyncMock()
        grok_provider.refresh_job = AsyncMock(return_value=None)

        service = UnifiedJobService(providers={Provider.GROK: grok_provider})  # type: ignore[arg-type]
        await service.get_job(job.id, user_id, session=session)

        grok_provider.refresh_job.assert_awaited_once_with(session, job)
