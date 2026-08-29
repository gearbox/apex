"""Tests for Grok generation provider request adaptation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import AsyncMock, MagicMock, call
from uuid import UUID, uuid4

from src.api.schemas.unified_generation import SourceMediaReference, UnifiedGenerationRequest
from src.api.services.generation.base import ProviderSubmitResult
from src.api.services.generation.grok_provider import GrokGenerationProvider
from src.api.services.generation.source_media import ResolvedSourceMedia
from src.core.enums import AspectRatio, GenerationType, JobStatus, MediaKind, ModelType
from src.core.library_ref import AssetRef, LibraryAssetSource, format_asset_ref

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.billing import BillingService
    from src.api.services.grok.job_service import GrokJobService
    from src.api.services.storage import R2StorageService


def _make_resolved_source(
    asset_id: UUID,
    *,
    source: LibraryAssetSource,
    storage_key: str,
) -> ResolvedSourceMedia:
    return ResolvedSourceMedia(
        position=0,
        ref=AssetRef(source=source, asset_id=asset_id),
        asset_ref=format_asset_ref(source, asset_id),
        media_kind=MediaKind.IMAGE,
        content_type="image/png",
        storage_key=storage_key,
        size_bytes=1,
        job_id=uuid4() if source is LibraryAssetSource.OUTPUT else None,
    )


async def test_grok_i2i_source_images_resolve_storage_references_to_provider_urls() -> None:
    user_id = uuid4()
    input_image_id = uuid4()
    source_output_id = uuid4()
    request = UnifiedGenerationRequest(
        prompt="use both image references",
        generation_type=GenerationType.I2I,
        model=ModelType.GROK_IMAGINE_IMAGE,
        n=2,
        aspect_ratio=AspectRatio.RATIO_9_16,
        source_media=[
            SourceMediaReference(asset_ref=f"upload:{input_image_id}"),
            SourceMediaReference(asset_ref=f"output:{source_output_id}"),
        ],
    )
    source_media = [
        _make_resolved_source(
            input_image_id,
            source=LibraryAssetSource.UPLOAD,
            storage_key="uploads/source.png",
        ),
        _make_resolved_source(
            source_output_id,
            source=LibraryAssetSource.OUTPUT,
            storage_key="outputs/source.webp",
        ),
    ]
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

    result = await provider.submit(
        request,
        user_id=user_id,
        session=cast("AsyncSession", AsyncMock()),
        billing_service=cast("BillingService", AsyncMock()),
        account_id=uuid4(),
        token_cost=25,
        product_id="vex",
        source_media=source_media,
    )

    assert result.balance_after == 88
    grok_job_service.create_image_job.assert_awaited_once()
    create_kwargs = grok_job_service.create_image_job.await_args.kwargs
    assert create_kwargs["input_image_url"] is None
    assert create_kwargs["input_image_urls"] == [
        "https://r2.test/uploads/source.png",
        "https://r2.test/outputs/source.webp",
    ]
    r2_storage.get_presigned_url.assert_has_awaits(
        [
            call("uploads/source.png", expires_in=3600),
            call("outputs/source.webp", expires_in=3600),
        ]
    )


async def test_grok_i2i_without_resolved_sources_skips_presigning() -> None:
    user_id = uuid4()
    request = UnifiedGenerationRequest(
        prompt="use an already-validated source",
        generation_type=GenerationType.I2I,
        model=ModelType.GROK_IMAGINE_IMAGE,
    )
    r2_storage = SimpleNamespace(get_presigned_url=AsyncMock())
    job = MagicMock(status=JobStatus.COMPLETED.value)
    grok_job_service = SimpleNamespace(
        create_image_job=AsyncMock(return_value=ProviderSubmitResult(job=job, balance_after=25))
    )
    provider = GrokGenerationProvider(
        cast("GrokJobService", grok_job_service),
        cast("R2StorageService", r2_storage),
    )

    await provider.submit(
        request,
        user_id=user_id,
        session=cast("AsyncSession", AsyncMock()),
        billing_service=cast("BillingService", AsyncMock()),
        account_id=uuid4(),
        token_cost=25,
        product_id="vex",
    )

    r2_storage.get_presigned_url.assert_not_awaited()
    create_kwargs = grok_job_service.create_image_job.await_args.kwargs
    assert create_kwargs["input_image_url"] is None
    assert create_kwargs["input_image_urls"] is None


async def test_grok_i2i_resolved_upload_is_presigned_for_provider() -> None:
    user_id = uuid4()
    input_image_id = uuid4()
    request = UnifiedGenerationRequest(
        prompt="edit uploaded image",
        generation_type=GenerationType.I2I,
        model=ModelType.GROK_IMAGINE_IMAGE,
        source_media=[SourceMediaReference(asset_ref=f"upload:{input_image_id}")],
    )
    source_media = [
        _make_resolved_source(
            input_image_id,
            source=LibraryAssetSource.UPLOAD,
            storage_key="uploads/source.png",
        )
    ]
    r2_storage = SimpleNamespace(
        get_presigned_url=AsyncMock(
            return_value=SimpleNamespace(presigned_url="https://r2.test/uploads/source.png")
        )
    )
    job = MagicMock()
    job.status = JobStatus.COMPLETED.value
    grok_job_service = SimpleNamespace(
        create_image_job=AsyncMock(return_value=ProviderSubmitResult(job=job, balance_after=77))
    )
    provider = GrokGenerationProvider(
        cast("GrokJobService", grok_job_service),
        cast("R2StorageService", r2_storage),
    )

    result = await provider.submit(
        request,
        user_id=user_id,
        session=cast("AsyncSession", AsyncMock()),
        billing_service=cast("BillingService", AsyncMock()),
        account_id=uuid4(),
        token_cost=25,
        product_id="vex",
        source_media=source_media,
    )

    assert result.balance_after == 77
    r2_storage.get_presigned_url.assert_awaited_once_with("uploads/source.png", expires_in=3600)
    create_kwargs = grok_job_service.create_image_job.await_args.kwargs
    assert create_kwargs["input_image_url"] == "https://r2.test/uploads/source.png"
    assert create_kwargs["input_image_urls"] is None
