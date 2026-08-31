"""Unit tests for the aisha-image-lite model (zit.cyberrealistic).

Covers the Part C contracts from agent_prompts/apex-aisha-image-lite.md:
Z-C3 (incompatible generation_type rejected before billing), Z-C4
(negative_prompt rejected for a model whose bundle has no negative_prompt
role), and Z-C5 (the same request succeeds when negative_prompt is omitted).

Model-registry-only contracts (Z-C1, Z-C2, Z-C6, Z-C8, Z-C9) live next to
the code they exercise: test_providers_endpoint.py, test_workflow_contract.py,
and test_bundle_index_service.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.rate_limiter import ModelRateLimiter
from src.api.services.generation.service import (
    GenerationService,
    UnsupportedGenerationParameterError,
)
from src.api.services.workflow.contract import BundleCapabilities
from src.core.enums import GenerationType, JobStatus, MediaKind, ModelType, Provider
from src.core.product_registry import VEX_CONFIG

pytestmark = pytest.mark.unit


def _make_mock_provider() -> MagicMock:
    provider = MagicMock()
    provider.validate = MagicMock()
    provider.supports_image_sizing = True
    job = MagicMock()
    job.id = uuid4()
    job.status = JobStatus.COMPLETED.value
    job.name = "Test"
    job.created_at = datetime.now(UTC)
    provider.submit = AsyncMock(
        return_value=ProviderSubmitResult(job=job, balance_after=100, balance_event=None)
    )
    return provider


def _make_enabled_model_mock() -> MagicMock:
    record = MagicMock()
    record.is_enabled = True
    return record


def _make_verified_user_mock() -> MagicMock:
    user = MagicMock()
    user.age_verified_at = datetime.now(UTC)
    return user


def _zit_capabilities() -> BundleCapabilities:
    """The mechanically-derived capability of the real zit.cyberrealistic
    bundle: t2i-only, no negative_prompt role (ConditioningZeroOut supplies
    the sampler's negative), 11 writable parameters."""
    return BundleCapabilities(
        media=MediaKind.IMAGE,
        generation_types=frozenset({GenerationType.T2I}),
        supports_negative_prompt=False,
        writable=frozenset(
            {
                "latent.width",
                "latent.height",
                "latent.batch_size",
                "positive_prompt.text",
                "sampler.seed",
                "sampler.steps",
                "sampler.cfg",
                "sampler.sampler",
                "sampler.scheduler",
                "sampler.denoise",
                "save.filename_prefix",
            }
        ),
        max_batch_size=4,
        max_reference_images=0,
    )


def _make_service(*, bundle_index: MagicMock | None = None) -> GenerationService:
    billing = AsyncMock()
    billing.resolve_account_for_user = AsyncMock(return_value=MagicMock(id=uuid4()))
    billing.assert_sufficient_balance = AsyncMock()
    billing.get_balance = AsyncMock(return_value=1000)

    pricing = AsyncMock()
    pricing.quote = AsyncMock(return_value=6)

    return GenerationService(
        providers={Provider.AISHA: _make_mock_provider()},  # type: ignore[dict-item]
        billing_service=billing,
        pricing_service=pricing,
        rate_limiter=MagicMock(spec=ModelRateLimiter),
        bundle_index=bundle_index,
    )


def _patched() -> tuple[Any, Any]:
    """Patch the DB lookups every generate() call needs, for an age-verified,
    enabled aisha-image-lite request — leaving only the assertion under test."""
    return (
        patch(
            "src.api.services.generation.service.GenerationModelRepository.get_by_model_key",
            new=AsyncMock(return_value=_make_enabled_model_mock()),
        ),
        patch(
            "src.db.repositories.user.UserRepository.get_user",
            new=AsyncMock(return_value=_make_verified_user_mock()),
        ),
    )


class TestAishaImageLiteBundleCapabilityValidation:
    async def test_i2i_request_rejected_before_billing(self) -> None:
        """Z-C3: aisha-image-lite's registry ceiling is t2i-only (matching the
        bundle exactly, per Z3 — "the intersection must not narrow something
        the operator believes is enabled"), so an i2i request is rejected at
        the registry-compatibility check, before the bundle index is even
        consulted. This mirrors the existing t2i-only pattern for
        ModelType.GROK_2_IMAGE (test_unified_generation.py::
        test_rejects_incompatible_model_generation_type)."""
        bundle_index = MagicMock()
        bundle_index.resolve_bundle.return_value = MagicMock(capabilities=_zit_capabilities())
        service = _make_service(bundle_index=bundle_index)
        request = UnifiedGenerationRequest(
            prompt="edit this",
            generation_type=GenerationType.I2I,
            model=ModelType.AISHA_IMAGE_LITE,
            input_image_id=uuid4(),
        )

        patch_model, patch_user = _patched()
        with patch_model, patch_user, pytest.raises(ValueError, match="does not support"):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        service._billing.assert_sufficient_balance.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_negative_prompt_rejected_for_a_bundle_with_no_negative_role(self) -> None:
        """Z-C4: zit.cyberrealistic has no negative_prompt role — a request
        that sets one is rejected with unsupported_generation_parameter,
        before billing."""
        bundle_index = MagicMock()
        bundle_index.resolve_bundle.return_value = MagicMock(capabilities=_zit_capabilities())
        service = _make_service(bundle_index=bundle_index)
        request = UnifiedGenerationRequest(
            prompt="a photo",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE_LITE,
            negative_prompt="blurry",
        )

        patch_model, patch_user = _patched()
        with (
            patch_model,
            patch_user,
            pytest.raises(UnsupportedGenerationParameterError) as error,
        ):
            await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        assert "negative_prompt" in error.value.parameters
        service._billing.assert_sufficient_balance.assert_not_awaited()  # type: ignore[attr-defined]

    async def test_negative_prompt_omitted_succeeds(self) -> None:
        """Z-C5: the same request without negative_prompt succeeds."""
        bundle_index = MagicMock()
        bundle_index.resolve_bundle.return_value = MagicMock(capabilities=_zit_capabilities())
        bundle_index.get_bound_workflow.return_value = None
        service = _make_service(bundle_index=bundle_index)
        request = UnifiedGenerationRequest(
            prompt="a photo",
            generation_type=GenerationType.T2I,
            model=ModelType.AISHA_IMAGE_LITE,
        )

        patch_model, patch_user = _patched()
        with patch_model, patch_user:
            result = await service.generate(
                request,
                user_id=uuid4(),
                session=AsyncMock(),
                product_config=VEX_CONFIG,
            )

        assert result.model == ModelType.AISHA_IMAGE_LITE.value
