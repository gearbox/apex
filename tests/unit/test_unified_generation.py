"""Tests for the unified generation endpoint and service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import msgspec
import pytest

from src.api.schemas.unified_generation import SourceImageReference, UnifiedGenerationRequest
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

    def test_source_images_accepts_storage_references(self) -> None:
        input_image_id = uuid4()
        source_output_id = uuid4()
        req = UnifiedGenerationRequest(
            prompt="Edit these",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_images=[
                SourceImageReference(input_image_id=input_image_id),
                SourceImageReference(source_output_id=source_output_id),
            ],
        )

        assert req.source_images is not None
        assert req.source_images[0].input_image_id == input_image_id
        assert req.source_images[1].source_output_id == source_output_id

    def test_source_images_allows_duplicate_references(self) -> None:
        input_image_id = uuid4()
        req = UnifiedGenerationRequest(
            prompt="Edit duplicates",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_images=[
                SourceImageReference(input_image_id=input_image_id),
                SourceImageReference(input_image_id=input_image_id),
            ],
        )

        assert req.source_images is not None
        assert [item.input_image_id for item in req.source_images] == [
            input_image_id,
            input_image_id,
        ]

    def test_source_images_storage_references_decode_from_json(self) -> None:
        input_image_id = uuid4()
        source_output_id = uuid4()
        payload = (
            b'{"prompt":"Edit these","generation_type":"i2i",'
            b'"model":"grok-imagine-image","source_images":['
            + f'{{"input_image_id":"{input_image_id}"}}'.encode()
            + b","
            + f'{{"source_output_id":"{source_output_id}"}}'.encode()
            + b"]}"
        )

        req = msgspec.json.decode(payload, type=UnifiedGenerationRequest)

        assert req.source_images is not None
        assert req.source_images[0].input_image_id == input_image_id
        assert req.source_images[1].source_output_id == source_output_id

    def test_source_images_rejects_more_than_four_references(self) -> None:
        refs = ",".join(f'{{"input_image_id":"{uuid4()}"}}' for _ in range(5))
        payload = (
            b'{"prompt":"Edit these","generation_type":"i2i",'
            b'"model":"grok-imagine-image","source_images":[' + refs.encode() + b"]}"
        )

        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(payload, type=UnifiedGenerationRequest)

    def test_source_images_reference_rejects_both_sources_on_decode(self) -> None:
        image_id = uuid4()
        output_id = uuid4()
        payload = (
            b'{"prompt":"Edit these","generation_type":"i2i",'
            b'"model":"grok-imagine-image","source_images":['
            + (f'{{"input_image_id":"{image_id}","source_output_id":"{output_id}"}}').encode()
            + b"]}"
        )

        with pytest.raises(msgspec.ValidationError):
            msgspec.json.decode(payload, type=UnifiedGenerationRequest)

    def test_source_images_reference_requires_exactly_one_source(self) -> None:
        image_id = uuid4()
        output_id = uuid4()

        with pytest.raises(ValueError, match="exactly one"):
            SourceImageReference(input_image_id=image_id, source_output_id=output_id)

        with pytest.raises(ValueError, match="exactly one"):
            SourceImageReference()

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
    pricing.quote = AsyncMock(return_value=50)

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

    def test_validate_i2i_accepts_source_images_references(self) -> None:
        service = _make_service()
        request = UnifiedGenerationRequest(
            prompt="Edit",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_images=[SourceImageReference(input_image_id=uuid4())],
        )

        service._validate_inputs(request)

    def test_validate_source_images_mutually_exclusive_with_scalar_inputs(self) -> None:
        service = _make_service()
        request = UnifiedGenerationRequest(
            prompt="Edit",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            input_image_id=uuid4(),
            source_images=[SourceImageReference(source_output_id=uuid4())],
        )

        with pytest.raises(ValueError, match="mutually exclusive"):
            service._validate_inputs(request)

    def test_input_image_count_t2i(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )

        assert GenerationService._input_image_count(request) == 0

    def test_input_image_count_input_image_id(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="Edit",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            input_image_id=uuid4(),
        )

        assert GenerationService._input_image_count(request) == 1

    def test_input_image_count_source_output_id(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="Edit",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_output_id=uuid4(),
        )

        assert GenerationService._input_image_count(request) == 1

    def test_input_image_count_source_images(self) -> None:
        request = UnifiedGenerationRequest(
            prompt="Edit",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_images=[
                SourceImageReference(input_image_id=uuid4()),
                SourceImageReference(input_image_id=uuid4()),
                SourceImageReference(input_image_id=uuid4()),
            ],
        )

        assert GenerationService._input_image_count(request) == 3


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
        pricing.quote = AsyncMock(return_value=50)

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
        pricing.quote = AsyncMock(return_value=50)

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
        pricing.quote = AsyncMock(return_value=50)

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
        pricing.quote = AsyncMock(return_value=50)

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
        pricing.quote = AsyncMock(return_value=50)

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

    async def test_generate_quotes_with_output_count_and_input_count(self) -> None:
        account = MagicMock(id=uuid4())
        mock_provider = _make_mock_provider()
        billing = AsyncMock()
        billing.resolve_account_for_user = AsyncMock(return_value=account)
        billing.assert_sufficient_balance = AsyncMock()
        pricing = AsyncMock()
        pricing.quote = AsyncMock(return_value=123)
        session = AsyncMock()

        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=billing,
            pricing_service=pricing,
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )

        request = UnifiedGenerationRequest(
            prompt="use uploads",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            n=2,
            source_images=[
                SourceImageReference(input_image_id=uuid4()),
                SourceImageReference(input_image_id=uuid4()),
                SourceImageReference(input_image_id=uuid4()),
            ],
        )
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.db.repositories.output.OutputRepository") as output_repo_cls,
        ):
            response = await service.generate(
                request,
                user_id=uuid4(),
                session=session,
                product_config=VEX_CONFIG,
            )

        output_repo_cls.assert_not_called()
        assert pricing.quote.await_args.kwargs["n"] == 2
        assert pricing.quote.await_args.kwargs["input_image_count"] == 3
        billing.assert_sufficient_balance.assert_awaited_once_with(account.id, 123, session=session)
        assert mock_provider.submit.await_args.kwargs["token_cost"] == 123
        assert response.tokens_charged == 123

    async def test_source_images_lineage_uses_first_output_reference(self) -> None:
        first_input_id = uuid4()
        first_output_id = uuid4()
        second_output_id = uuid4()
        source_job_id = uuid4()
        user_id = uuid4()
        source_output = MagicMock()
        source_output.job_id = source_job_id
        output_repo = MagicMock()
        output_repo.get = AsyncMock(return_value=source_output)
        mock_provider = _make_mock_provider()
        service = _make_service(providers={Provider.GROK: mock_provider})
        session = AsyncMock()
        request = UnifiedGenerationRequest(
            prompt="use several references",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_images=[
                SourceImageReference(input_image_id=first_input_id),
                SourceImageReference(source_output_id=first_output_id),
                SourceImageReference(source_output_id=second_output_id),
            ],
        )

        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.db.repositories.output.OutputRepository", return_value=output_repo),
        ):
            await service.generate(
                request,
                user_id=user_id,
                session=session,
                product_config=VEX_CONFIG,
            )

        output_repo.get.assert_awaited_once_with(first_output_id, user_id=user_id)
        submit_kwargs = mock_provider.submit.await_args.kwargs
        assert submit_kwargs["source_job_id"] == source_job_id
        assert submit_kwargs["source_output_id"] == first_output_id

    async def test_source_images_all_input_references_have_no_lineage(self) -> None:
        mock_provider = _make_mock_provider()
        service = _make_service(providers={Provider.GROK: mock_provider})
        request = UnifiedGenerationRequest(
            prompt="use uploads",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_images=[
                SourceImageReference(input_image_id=uuid4()),
                SourceImageReference(input_image_id=uuid4()),
            ],
        )

        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.db.repositories.output.OutputRepository") as output_repo_cls,
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        output_repo_cls.assert_not_called()
        submit_kwargs = mock_provider.submit.await_args.kwargs
        assert submit_kwargs["source_job_id"] is None
        assert submit_kwargs["source_output_id"] is None

    async def test_source_images_output_reference_not_owned_raises_value_error(self) -> None:
        source_output_id = uuid4()
        user_id = uuid4()
        output_repo = MagicMock()
        output_repo.get = AsyncMock(return_value=None)
        mock_provider = _make_mock_provider()
        service = _make_service(providers={Provider.GROK: mock_provider})
        request = UnifiedGenerationRequest(
            prompt="use output",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_images=[SourceImageReference(source_output_id=source_output_id)],
        )

        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.db.repositories.output.OutputRepository", return_value=output_repo),
            pytest.raises(ValueError, match="Source output not found"),
        ):
            await service.generate(
                request,
                user_id=user_id,
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        output_repo.get.assert_awaited_once_with(source_output_id, user_id=user_id)
        mock_provider.submit.assert_not_awaited()


class TestGenerationServiceSlidingRetention:
    """Sliding upload retention: using an upload as input resets its expiry."""

    async def test_generate_bumps_input_image_expiry(self) -> None:
        input_image_id = uuid4()
        user_id = uuid4()
        mock_provider = _make_mock_provider()
        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=AsyncMock(
                resolve_account_for_user=AsyncMock(return_value=MagicMock(id=uuid4())),
                assert_sufficient_balance=AsyncMock(),
            ),
            pricing_service=AsyncMock(quote=AsyncMock(return_value=50)),
            rate_limiter=MagicMock(spec=ModelRateLimiter),
            retention_days=7,
        )
        request = UnifiedGenerationRequest(
            prompt="Edit this",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            input_image_id=input_image_id,
        )

        image_repo = MagicMock()
        image_repo.touch_expiry = AsyncMock(return_value=True)
        before = datetime.now(UTC)
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch(
                "src.api.services.generation.service.UserImageRepository",
                return_value=image_repo,
            ),
        ):
            await service.generate(
                request,
                user_id=user_id,
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )
        after = datetime.now(UTC)

        image_repo.touch_expiry.assert_awaited_once()
        call_kwargs = image_repo.touch_expiry.await_args.kwargs
        assert image_repo.touch_expiry.await_args.args[0] == input_image_id
        assert call_kwargs["user_id"] == user_id
        expected_min = before + timedelta(days=7) - timedelta(seconds=5)
        expected_max = after + timedelta(days=7) + timedelta(seconds=5)
        assert expected_min <= call_kwargs["expires_at"] <= expected_max
        # Called before provider.submit
        mock_provider.submit.assert_awaited_once()

    async def test_generate_unowned_input_image_rejected(self) -> None:
        input_image_id = uuid4()
        mock_provider = _make_mock_provider()
        service = GenerationService(
            providers={Provider.GROK: mock_provider},
            billing_service=AsyncMock(
                resolve_account_for_user=AsyncMock(return_value=MagicMock(id=uuid4())),
                assert_sufficient_balance=AsyncMock(),
            ),
            pricing_service=AsyncMock(quote=AsyncMock(return_value=50)),
            rate_limiter=MagicMock(spec=ModelRateLimiter),
        )
        request = UnifiedGenerationRequest(
            prompt="Edit this",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            input_image_id=input_image_id,
        )

        image_repo = MagicMock()
        image_repo.touch_expiry = AsyncMock(return_value=False)
        session = AsyncMock()
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch(
                "src.api.services.generation.service.UserImageRepository",
                return_value=image_repo,
            ),
            pytest.raises(ValueError, match="Input image not found"),
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=session,
                product_config=VEX_CONFIG,
            )

        mock_provider.submit.assert_not_awaited()
        session.commit.assert_not_awaited()

    async def test_generate_source_output_does_not_touch_uploads(self) -> None:
        source_output_id = uuid4()
        user_id = uuid4()
        source_output = MagicMock()
        source_output.job_id = uuid4()
        output_repo = MagicMock()
        output_repo.get = AsyncMock(return_value=source_output)
        mock_provider = _make_mock_provider()
        service = _make_service(providers={Provider.GROK: mock_provider})
        request = UnifiedGenerationRequest(
            prompt="Remix",
            generation_type=GenerationType.I2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
            source_output_id=source_output_id,
        )

        image_repo = MagicMock()
        image_repo.touch_expiry = AsyncMock(return_value=True)
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch("src.db.repositories.output.OutputRepository", return_value=output_repo),
            patch(
                "src.api.services.generation.service.UserImageRepository",
                return_value=image_repo,
            ),
        ):
            await service.generate(
                request,
                user_id=user_id,
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        image_repo.touch_expiry.assert_not_awaited()

    async def test_generate_t2i_does_not_touch_uploads(self) -> None:
        mock_provider = _make_mock_provider()
        service = _make_service(providers={Provider.GROK: mock_provider})
        request = UnifiedGenerationRequest(
            prompt="A cat",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )

        image_repo = MagicMock()
        image_repo.touch_expiry = AsyncMock(return_value=True)
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch(
                "src.api.services.generation.service.UserImageRepository",
                return_value=image_repo,
            ),
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        image_repo.touch_expiry.assert_not_awaited()


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
        mock_pricing.quote = AsyncMock(return_value=50)

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
