"""Tests for the unified generation endpoint and service."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import (
    GenerationError,
    GenerationService,
    ModelDisabledError,
    ProviderUnavailableError,
)
from src.core.enums import (
    AspectRatio,
    GenerationType,
    JobStatus,
    ModelType,
    Provider,
    Resolution,
    Sampler,
    Scheduler,
    VideoResolution,
)
from src.core.product_registry import VEX_CONFIG

# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------


class TestUnifiedGenerationRequestSchema:
    def test_minimal_valid_t2i(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        assert req.prompt == "A cat"
        assert req.n == 1
        assert req.aspect_ratio == AspectRatio.RATIO_1_1
        assert req.input_image_id is None

    def test_i2i_without_image_id_is_valid_at_schema_level(self) -> None:
        """Schema does not enforce input_image_id — the service does."""
        req = UnifiedGenerationRequest(
            prompt="Edit this",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        assert req.input_image_id is None

    def test_video_fields(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="A sunset timelapse",
            generation_type=GenerationType.T2V,
            model=ModelType.GROK_IMAGINE_VIDEO,
            duration=10,
            resolution=VideoResolution.RES_480P,
            aspect_ratio=AspectRatio.RATIO_16_9,
        )
        assert req.duration == 10
        assert req.resolution == VideoResolution.RES_480P

    def test_aisha_specific_fields(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="A landscape",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            image_resolution=Resolution.STANDARD,
            seed=42,
            steps=8,
        )
        assert req.seed == 42
        assert req.steps == 8

    def test_aisha_explicit_width_and_height(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="A landscape",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            width=768,
            height=768,
        )
        assert req.width == 768
        assert req.height == 768

    def test_forbid_unknown_fields(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","bogus":true}',
                type=UnifiedGenerationRequest,
            )

    def test_prompt_validation(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"","generation_type":"t2i","model":"aisha-image"}',
                type=UnifiedGenerationRequest,
            )

    def test_n_validation_bounds(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","n":0}',
                type=UnifiedGenerationRequest,
            )

    def test_json_roundtrip(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="test",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
        )
        encoded = msgspec.json.encode(req)
        decoded = msgspec.json.decode(encoded, type=UnifiedGenerationRequest)
        assert decoded.prompt == "test"


# ---------------------------------------------------------------------------
# Image resolution + sampler field tests
# ---------------------------------------------------------------------------


class TestImageResolutionFields:
    def test_image_resolution_ultra_accepted(self) -> None:
        req = msgspec.json.decode(
            b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","image_resolution":"ultra"}',
            type=UnifiedGenerationRequest,
        )
        assert req.image_resolution == Resolution.ULTRA

    def test_width_and_height_together_accepted(self) -> None:
        req = msgspec.json.decode(
            b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","width":512,"height":512}',
            type=UnifiedGenerationRequest,
        )
        assert req.width == 512
        assert req.height == 512

    def test_only_width_no_height_raises(self) -> None:
        with pytest.raises((msgspec.ValidationError, ValueError)):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","width":512}',
                type=UnifiedGenerationRequest,
            )

    def test_only_height_no_width_raises(self) -> None:
        with pytest.raises((msgspec.ValidationError, ValueError)):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","height":512}',
                type=UnifiedGenerationRequest,
            )

    def test_image_resolution_and_explicit_dims_mutually_exclusive(self) -> None:
        with pytest.raises((msgspec.ValidationError, ValueError)):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image",'
                b'"image_resolution":"standard","width":512,"height":512}',
                type=UnifiedGenerationRequest,
            )

    def test_video_resolution_720p_still_accepted(self) -> None:
        req = msgspec.json.decode(
            b'{"prompt":"x","generation_type":"t2v","model":"grok-imagine-video","resolution":"720p"}',
            type=UnifiedGenerationRequest,
        )
        assert req.resolution == VideoResolution.RES_720P

    def test_video_resolution_1080p_rejected(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2v","model":"grok-imagine-video","resolution":"1080p"}',
                type=UnifiedGenerationRequest,
            )


class TestSamplerFields:
    def test_cfg_accepts_valid_value(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="x",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            cfg=7.5,
        )
        assert req.cfg == pytest.approx(7.5)

    def test_sampler_accepts_valid_value(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="x",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            sampler=Sampler.DPMPP_2M,
        )
        assert req.sampler == Sampler.DPMPP_2M

    def test_scheduler_accepts_valid_value(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="x",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            scheduler=Scheduler.KARRAS,
        )
        assert req.scheduler == Scheduler.KARRAS

    def test_denoise_accepts_valid_value(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="x",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            denoise=0.75,
        )
        assert req.denoise == pytest.approx(0.75)

    def test_invalid_sampler_enum_rejected(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","sampler":"not_real"}',
                type=UnifiedGenerationRequest,
            )

    def test_steps_up_to_150_accepted(self) -> None:
        req = UnifiedGenerationRequest(
            prompt="x",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            steps=150,
        )
        assert req.steps == 150

    def test_steps_above_150_rejected(self) -> None:
        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(
                b'{"prompt":"x","generation_type":"t2i","model":"aisha-image","steps":151}',
                type=UnifiedGenerationRequest,
            )


# ---------------------------------------------------------------------------
# GenerationService tests
# ---------------------------------------------------------------------------


def _make_mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.validate = MagicMock()
    job = MagicMock()
    job.id = uuid4()
    job.status = JobStatus.COMPLETED.value
    job.name = "Test"
    job.created_at = datetime.now(UTC)
    provider.submit = AsyncMock(
        return_value=ProviderSubmitResult(job=job, balance_after=100, balance_event=None)
    )
    return provider


def _make_service(
    providers: dict | None = None,
) -> GenerationService:
    billing = AsyncMock()
    billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
    billing.assert_sufficient_balance = AsyncMock()
    billing.get_balance = AsyncMock(return_value=1000)

    pricing = AsyncMock()
    pricing.get_price = AsyncMock(return_value=50)

    if providers is None:
        providers = {Provider.GROK: _make_mock_provider()}

    return GenerationService(
        providers=providers,  # type: ignore[arg-type]
        billing_service=billing,
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
    )


class TestGenerationServiceValidation:
    def test_validate_i2i_requires_image_id(self) -> None:
        service = _make_service()
        request = UnifiedGenerationRequest(
            prompt="Edit",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        with pytest.raises(ValueError, match="requires input_image_id"):
            service._validate_inputs(request)

    def test_validate_v2v_requires_video_url(self) -> None:
        service = _make_service()
        request = UnifiedGenerationRequest(
            prompt="Edit video",
            generation_type=GenerationType.V2V,
            model=ModelType.GROK_IMAGINE_VIDEO,
        )
        with pytest.raises(ValueError, match="requires input_video_url"):
            service._validate_inputs(request)

    def test_validate_t2i_passes(self) -> None:
        service = _make_service()
        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        service._validate_inputs(request)  # Should not raise


def _make_enabled_model_mock() -> MagicMock:
    model_record = MagicMock()
    model_record.is_enabled = True
    return model_record


class TestGenerationServiceGenerate:
    async def test_raises_on_missing_provider(self) -> None:
        service = _make_service(providers={})
        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            pytest.raises((ProviderUnavailableError, ModelDisabledError)),
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

    async def test_rejects_incompatible_model_generation_type(self) -> None:
        """grok-2-image does not support I2I — should fail before reaching provider."""
        service = _make_service()
        request = UnifiedGenerationRequest(
            prompt="Edit",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_2_IMAGE,
            input_image_id=uuid4(),
        )
        with pytest.raises(ValueError, match="does not support"):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

    async def test_rejects_n_exceeding_model_cap(self) -> None:
        """Aisha supports max 4 outputs."""
        service = _make_service(providers={Provider.AISHA: _make_mock_provider()})  # type: ignore[arg-type]
        request = UnifiedGenerationRequest(
            prompt="Cats",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
            n=8,
        )
        with pytest.raises(ValueError, match="max 4 outputs"):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

    async def test_generic_provider_error_rolls_back_without_leaking_message(self) -> None:
        """A bare Exception from provider.submit() must not leak into the
        response — only the fixed generic message may. Since submit() raised
        before returning a job, nothing durable was written: the session
        must roll back (there is no job to refund, and nothing to commit)."""
        mock_provider = _make_mock_provider()
        mock_provider.submit = AsyncMock(side_effect=Exception("tunnel host 10.0.0.5 unreachable"))

        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing.assert_sufficient_balance = AsyncMock()
        billing.refund = AsyncMock()

        pricing = AsyncMock()
        pricing.get_price = AsyncMock(return_value=50)

        session = AsyncMock()

        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )

        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            pytest.raises(GenerationError) as exc_info,
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=session,
                product_config=VEX_CONFIG,
            )

        # Fixed, generic message only — the real exception text must not leak.
        assert str(exc_info.value) == "Generation failed due to an internal error."
        assert "tunnel host" not in str(exc_info.value)

        # submit() never returned a job, so there is nothing to refund —
        # discard the (empty) transaction instead of committing a charge.
        billing.refund.assert_not_awaited()
        session.rollback.assert_awaited()

    async def test_asyncpg_style_db_error_does_not_leak_into_response(self) -> None:
        """A raw DB-driver-style exception (e.g. asyncpg.PostgresError, or
        SQLAlchemy wrapping one) must never reach the client — only the fixed
        generic GenerationError message may."""

        class FakeUniqueViolationError(Exception):
            """Stands in for asyncpg.exceptions.UniqueViolationError."""

        mock_provider = _make_mock_provider()
        mock_provider.submit = AsyncMock(
            side_effect=FakeUniqueViolationError(
                'duplicate key value violates unique constraint "ix_generation_jobs_external_request_id" '
                "DETAIL:  Key (external_request_id)=(prompt-abc123) already exists."
            )
        )

        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing.assert_sufficient_balance = AsyncMock()
        billing.refund = AsyncMock()

        pricing = AsyncMock()
        pricing.get_price = AsyncMock(return_value=50)

        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )

        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            pytest.raises(GenerationError) as exc_info,
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        message = str(exc_info.value)
        assert message == "Generation failed due to an internal error."
        assert "constraint" not in message
        assert "external_request_id" not in message
        assert "DETAIL" not in message

    async def test_commit_failure_after_submit_refunds_and_rolls_back(self) -> None:
        """If provider.submit() succeeds (job + debit written) but the
        subsequent session.commit() fails, the orchestrator must refund the
        job and commit that compensating transaction — never leave a debit
        on the ledger with no refund."""
        job = MagicMock()
        job.id = uuid4()
        job.status = JobStatus.QUEUED.value
        job.name = "Test"
        job.created_at = datetime.now(UTC)

        mock_provider = MagicMock()
        mock_provider.validate = MagicMock()
        mock_provider.submit = AsyncMock(
            return_value=ProviderSubmitResult(job=job, balance_after=100, balance_event=None)
        )

        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing.assert_sufficient_balance = AsyncMock()
        billing.refund = AsyncMock(return_value=MagicMock(event=None))

        pricing = AsyncMock()
        pricing.get_price = AsyncMock(return_value=50)

        session = AsyncMock()
        # First commit() (right after submit succeeds) fails; the recovery
        # commit inside the failure-handling path should still succeed.
        session.commit = AsyncMock(side_effect=[Exception("db connection lost"), None])

        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )

        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            pytest.raises(GenerationError) as exc_info,
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=session,
                product_config=VEX_CONFIG,
            )

        assert str(exc_info.value) == "Generation failed due to an internal error."
        billing.refund.assert_awaited_once()
        assert billing.refund.await_args.args[0] == job.id
        assert session.commit.await_count == 2

    async def test_job_status_publish_failure_does_not_refund_or_error(self) -> None:
        """A post-commit publish() failure (e.g. a Redis hiccup on the
        JOB_STATUS_CHANGED event) must never refund an already-committed,
        already-billed job or surface as a client-facing error (R3)."""
        job = MagicMock()
        job.id = uuid4()
        job.status = JobStatus.QUEUED.value
        job.name = "Test"
        job.created_at = datetime.now(UTC)

        mock_provider = MagicMock()
        mock_provider.validate = MagicMock()
        mock_provider.supports_image_sizing = True
        mock_provider.submit = AsyncMock(
            return_value=ProviderSubmitResult(job=job, balance_after=950, balance_event=None)
        )

        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing.assert_sufficient_balance = AsyncMock()
        billing.refund = AsyncMock()

        pricing = AsyncMock()
        pricing.get_price = AsyncMock(return_value=50)

        event_bus = AsyncMock()
        event_bus.publish_balance = AsyncMock()
        event_bus.publish = AsyncMock(side_effect=Exception("redis unavailable"))

        session = AsyncMock()

        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
            event_bus=event_bus,
        )

        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.api.services.generation.service.logger") as mock_logger,
        ):
            response = await service.generate(
                request,
                user_id=uuid4(),
                session=session,
                product_config=VEX_CONFIG,
            )

        assert response.job_id == job.id
        assert response.balance_remaining == 950
        billing.refund.assert_not_awaited()
        session.rollback.assert_not_awaited()
        mock_logger.exception.assert_called_once()
        assert mock_logger.exception.call_args.args[0] == "generation.post_commit_publish_failed"

    async def test_response_balance_from_submit_result_no_extra_ledger_scan(self) -> None:
        """``balance_remaining`` must come from ``ProviderSubmitResult.balance_after``
        — no extra ``get_balance()`` ledger scan after a successful submit (R6/R7)."""
        job = MagicMock()
        job.id = uuid4()
        job.status = JobStatus.COMPLETED.value
        job.name = "Test"
        job.created_at = datetime.now(UTC)

        mock_provider = MagicMock()
        mock_provider.validate = MagicMock()
        mock_provider.supports_image_sizing = True
        mock_provider.submit = AsyncMock(
            return_value=ProviderSubmitResult(job=job, balance_after=725, balance_event=None)
        )

        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        billing.assert_sufficient_balance = AsyncMock()

        pricing = AsyncMock()
        pricing.get_price = AsyncMock(return_value=50)

        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )

        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        with patch(
            "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
            new=AsyncMock(return_value=_make_enabled_model_mock()),
        ):
            response = await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        assert response.balance_remaining == 725
        billing.get_balance.assert_not_called()


class TestGenerationServiceRateLimit:
    """Test rate limit integration in GenerationService."""

    async def test_rate_limit_exceeded_raises(self) -> None:
        """GenerationService should propagate RateLimitExceededError."""
        from src.api.services.generation.rate_limiter import RateLimitExceededError

        mock_limiter = MagicMock(spec=ModelRateLimiter)
        mock_limiter.check.side_effect = RateLimitExceededError(
            model=ModelType.GROK_IMAGINE_VIDEO,
            retry_after=30,
        )

        mock_model_record = MagicMock()
        mock_model_record.is_enabled = True

        mock_session = AsyncMock()
        mock_providers: dict = {Provider.GROK: _make_mock_provider()}
        mock_billing = AsyncMock()
        mock_billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
        mock_billing.assert_sufficient_balance = AsyncMock()
        mock_billing.get_balance = AsyncMock(return_value=1000)
        mock_pricing = AsyncMock()
        mock_pricing.get_price = AsyncMock(return_value=50)

        service = GenerationService(
            providers=mock_providers,
            billing_service=mock_billing,
            pricing_service=mock_pricing,
            rate_limiter=mock_limiter,
        )

        request = UnifiedGenerationRequest(
            prompt="Test",
            generation_type=GenerationType.T2V,
            model=ModelType.GROK_IMAGINE_VIDEO,
        )

        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=mock_model_record),
            ),
            pytest.raises(RateLimitExceededError),
        ):
            await service.generate(
                request, user_id=uuid4(), session=mock_session, product_config=VEX_CONFIG
            )


# ---------------------------------------------------------------------------
# UnifiedJobService — poll-on-read unknown provider guard (R4)
# ---------------------------------------------------------------------------


class TestUnifiedJobServiceUnknownProvider:
    """An unrecognized ``job.provider`` value must not crash GET /v1/jobs/{id}."""

    async def test_get_job_unknown_provider_serves_stale_row(self) -> None:
        from src.api.services.unified_jobs import UnifiedJobService

        job_id = uuid4()
        job = MagicMock()
        job.id = job_id
        job.provider = "some-retired-provider"
        job.status = JobStatus.COMPLETED.value
        job.generation_type = GenerationType.T2I.value
        job.outputs = []

        with (
            patch("src.api.services.unified_jobs.JobRepository") as job_repo_cls,
            patch("src.api.services.unified_jobs.OutputRepository") as output_repo_cls,
        ):
            job_repo_cls.return_value.get = AsyncMock(return_value=job)
            output_repo_cls.return_value.list_by_job = AsyncMock(return_value=[])
            output_repo_cls.return_value.batch_derivatives = AsyncMock(return_value={})

            service = UnifiedJobService(providers={})
            result = await service.get_job(job_id, uuid4(), session=AsyncMock())

        assert result is not None
        assert result.id == job_id
        assert result.provider == "some-retired-provider"
