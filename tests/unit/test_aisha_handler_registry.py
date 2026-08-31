"""Contracts for the Aisha media-handler facade."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.schemas.unified_generation import UnifiedGenerationRequest
from src.api.services.bundle_index import BundleNotFoundError
from src.api.services.generation.aisha.handlers import UnsupportedMediaError
from src.api.services.generation.aisha_provider import AishaGenerationProvider
from src.api.services.generation.base import ProviderSubmitResult
from src.core.enums import GenerationType, JobStatus, MediaKind, ModelType


class _ImageHandler:
    media = MediaKind.IMAGE

    def __init__(self) -> None:
        self.validate_calls = 0
        self.submit = AsyncMock()

    def validate(self, _request: object, _capabilities: object) -> None:
        self.validate_calls += 1

    async def prepare_inputs(self, _source_media: object) -> dict[object, list[str]]:
        return {}


def _request(generation_type: GenerationType = GenerationType.T2I) -> UnifiedGenerationRequest:
    return UnifiedGenerationRequest(
        prompt="a fox",
        generation_type=generation_type,
        model=ModelType.AISHA_IMAGE,
    )


async def test_submit_delegates_to_the_handler_for_output_media() -> None:
    handler = _ImageHandler()
    job = MagicMock(status=JobStatus.PENDING)
    handler.submit.return_value = ProviderSubmitResult(job=job, balance_after=42)
    provider = AishaGenerationProvider(
        workflow_service=MagicMock(),
        gpu_session_service=None,
        bundle_index=MagicMock(),
        handlers={MediaKind.IMAGE: handler},  # type: ignore[dict-item]
    )

    result = await provider.submit(
        _request(),
        user_id=uuid4(),
        session=AsyncMock(),
        billing_service=AsyncMock(),
        account_id=uuid4(),
        token_cost=1,
        product_id="vex",
    )

    assert result.balance_after == 42
    handler.submit.assert_awaited_once()


def test_unregistered_output_media_names_the_media_not_a_model() -> None:
    provider = AishaGenerationProvider(
        workflow_service=MagicMock(),
        gpu_session_service=None,
        bundle_index=MagicMock(),
        handlers={},
    )

    with pytest.raises(UnsupportedMediaError, match="video media"):
        provider.validate(_request(GenerationType.T2V))


def test_missing_indexed_bundle_has_no_capabilities() -> None:
    """An unindexed model stays disabled instead of making validation fail."""
    bundle_index = MagicMock()
    bundle_index.resolve_bundle.side_effect = BundleNotFoundError("not indexed")
    provider = AishaGenerationProvider(
        workflow_service=MagicMock(),
        gpu_session_service=None,
        bundle_index=bundle_index,
        handlers={},
    )

    assert provider._capabilities_for(ModelType.AISHA_IMAGE_LITE) is None
    bundle_index.resolve_bundle.assert_called_once_with(ModelType.AISHA_IMAGE_LITE.value)
