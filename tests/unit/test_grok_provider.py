"""Tests for Grok generation provider request adaptation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

from src.api.schemas.unified_generation import SourceImageReference, UnifiedGenerationRequest
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.grok_provider import GrokGenerationProvider
from src.core.enums import AspectRatio, GenerationType, JobStatus, ModelType

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BillingService
    from src.api.services.grok.job_service import GrokJobService
    from src.api.services.storage import R2StorageService


async def test_grok_i2i_source_images_resolve_storage_references_to_provider_urls() -> None:
    input_image_id = uuid4()
    source_output_id = uuid4()
    request = UnifiedGenerationRequest(
        prompt="use both image references",
        generation_type=GenerationType.I2I,
        model=ModelType.GROK_IMAGINE_IMAGE,
        n=2,
        aspect_ratio=AspectRatio.RATIO_9_16,
        source_images=[
            SourceImageReference(input_image_id=input_image_id),
            SourceImageReference(source_output_id=source_output_id),
        ],
    )
    user_image_repo = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(storage_key="uploads/source.png"))
    )
    output_repo = SimpleNamespace(
        get=AsyncMock(return_value=SimpleNamespace(storage_key="outputs/source.webp"))
    )
    r2_storage = SimpleNamespace(
        get_presigned_url=AsyncMock(
            side_effect=[
                SimpleNamespace(presigned_url="https://r2.test/uploads/source.png"),
                SimpleNamespace(presigned_url="https://r2.test/outputs/source.webp"),
            ]
        )
    )
    job = MagicMock()
    job.status = JobStatus.COMPLETED.value
    grok_job_service = SimpleNamespace(
        create_image_job=AsyncMock(return_value=ProviderSubmitResult(job=job, balance_after=88))
    )
    provider = GrokGenerationProvider(
        cast("GrokJobService", grok_job_service),
        cast("R2StorageService", r2_storage),
    )

    with (
        patch(
            "src.db.repositories.user_image.UserImageRepository",
            return_value=user_image_repo,
        ),
        patch("src.db.repositories.output.OutputRepository", return_value=output_repo),
    ):
        result = await provider.submit(
            request,
            user_id=uuid4(),
            session=cast("AsyncSession", AsyncMock()),
            billing_service=cast("BillingService", AsyncMock()),
            account_id=uuid4(),
            token_cost=25,
            product_id="vex",
        )

    assert result.balance_after == 88
    grok_job_service.create_image_job.assert_awaited_once()
    create_kwargs = grok_job_service.create_image_job.await_args.kwargs
    assert create_kwargs["input_image_url"] is None
    assert create_kwargs["input_image_urls"] == [
        "https://r2.test/uploads/source.png",
        "https://r2.test/outputs/source.webp",
    ]
    user_image_repo.get.assert_awaited_once_with(input_image_id)
    output_repo.get.assert_awaited_once_with(source_output_id)
