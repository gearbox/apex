"""Unit tests for the per-model age-verification gate.

Covers:
- ModelType.requires_age_verification property
- GenerationService.generate() gate behaviour
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import (
    AgeVerificationRequiredError,
    GenerationService,
)
from src.core.enums import GenerationType, JobStatus, ModelType, Provider

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# ModelType.requires_age_verification property
# ---------------------------------------------------------------------------


class TestModelTypeRequiresAgeVerification:
    @pytest.mark.parametrize("mt", [ModelType.AISHA_IMAGE, ModelType.AISHA_VIDEO])
    def test_aisha_models_require_age_verification(self, mt: ModelType) -> None:
        assert mt.requires_age_verification is True

    @pytest.mark.parametrize(
        "mt",
        [
            ModelType.GROK_IMAGINE_IMAGE,
            ModelType.GROK_2_IMAGE,
            ModelType.GROK_IMAGINE_VIDEO,
        ],
    )
    def test_grok_models_do_not_require_age_verification(self, mt: ModelType) -> None:
        assert mt.requires_age_verification is False


# ---------------------------------------------------------------------------
# GenerationService.generate() — age gate
# ---------------------------------------------------------------------------


def _make_mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.validate = MagicMock()
    job = MagicMock()
    job.id = uuid4()
    job.status = JobStatus.COMPLETED.value
    job.name = "Test"
    job.created_at = datetime.now(UTC)
    provider.submit = AsyncMock(return_value=job)
    return provider


def _make_enabled_model_mock() -> MagicMock:
    record = MagicMock()
    record.is_enabled = True
    return record


def _make_billing_mock() -> AsyncMock:
    billing = AsyncMock()
    billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
    billing.assert_sufficient_balance = AsyncMock()
    billing.get_balance = AsyncMock(return_value=1000)
    return billing


def _make_generation_service(provider: Provider = Provider.AISHA) -> GenerationService:
    pricing = AsyncMock()
    pricing.get_price = AsyncMock(return_value=50)
    return GenerationService(
        providers={provider: _make_mock_provider()},  # type: ignore[dict-item]
        billing_service=_make_billing_mock(),
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
    )


def _make_user_mock(*, age_verified_at: datetime | None) -> MagicMock:
    u = MagicMock()
    u.age_verified_at = age_verified_at
    return u


class TestGenerationAgeGate:
    async def test_unverified_user_aisha_image_raises(self) -> None:
        service = _make_generation_service()
        request = UnifiedGenerationRequest(
            prompt="test",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
        )
        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            # UserRepository is imported lazily; patch at the source module
            patch(
                "src.db.repositories.user.UserRepository.get_user",
                new=AsyncMock(return_value=_make_user_mock(age_verified_at=None)),
            ),
            pytest.raises(AgeVerificationRequiredError),
        ):
            await service.generate(request, user_id=uuid4(), session=AsyncMock())

    async def test_verified_user_aisha_image_passes_gate(self) -> None:
        service = _make_generation_service()
        request = UnifiedGenerationRequest(
            prompt="test",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE,
        )
        verified_at = datetime.now(UTC)
        session_mock = AsyncMock()

        with (
            patch(
                "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
                new=AsyncMock(return_value=_make_enabled_model_mock()),
            ),
            patch(
                "src.db.repositories.user.UserRepository.get_user",
                new=AsyncMock(return_value=_make_user_mock(age_verified_at=verified_at)),
            ),
        ):
            # Should pass the gate (and complete the full pipeline with mocks)
            result = await service.generate(request, user_id=uuid4(), session=session_mock)

        assert result.model == ModelType.AISHA_IMAGE.value

    async def test_unverified_user_grok_image_not_gated(self) -> None:
        """Grok models never hit the age gate regardless of user verification status."""
        service = _make_generation_service(Provider.GROK)
        request = UnifiedGenerationRequest(
            prompt="test",
            generation_type=GenerationType.T2I,
            model=ModelType.GROK_IMAGINE_IMAGE,
        )
        session_mock = AsyncMock()

        with patch(
            "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
            new=AsyncMock(return_value=_make_enabled_model_mock()),
        ):
            # Must complete without raising AgeVerificationRequiredError
            result = await service.generate(request, user_id=uuid4(), session=session_mock)

        assert result.model == ModelType.GROK_IMAGINE_IMAGE.value
