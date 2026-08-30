"""Grok generation provider — adapter for GrokJobService."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

import structlog

from src.core.enums import GenerationType, JobStatus, MediaKind

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.api.services.generation.base import ProviderSubmitResult
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.api.services.grok.job_service import GrokJobService
    from src.api.services.storage import R2StorageService
    from src.db.models.storage import GenerationJob

logger = structlog.get_logger(__name__)

_PRESIGN_EXPIRES_SECONDS = 3600


class GrokGenerationProvider:
    """Adapts GrokJobService to the GenerationProvider protocol.

    Model-generation_type compatibility and n-cap validation are handled
    by the GenerationService orchestrator via ModelType enum properties.
    This class only contains Grok-specific submission logic.
    """

    # Grok ignores request-level image sizing fields — its outputs use
    # fixed sizes per aspect ratio/resolution, not resolve_dimensions().
    supports_image_sizing: ClassVar[bool] = False

    def __init__(
        self,
        grok_job_service: GrokJobService,
        r2_storage: R2StorageService,
    ) -> None:
        self._grok = grok_job_service
        self._r2 = r2_storage

    def validate(self, request: UnifiedGenerationRequest) -> None:
        """Grok-specific validation beyond what the enum provides.

        Currently a no-op — all model-generation_type and n-cap checks
        are handled by the orchestrator.
        """

    async def refresh_job(
        self,
        session: AsyncSession,
        job: GenerationJob,
    ) -> GenerationJob | None:
        """Poll xAI for a fresh status on queued/running async video jobs.

        No-op for image jobs (synchronous — always terminal by the time the
        row exists) and for jobs already in a terminal status.
        """
        if GenerationType(job.generation_type).output_kind is not MediaKind.VIDEO:
            return None
        if job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return None
        return await self._grok.poll_video_job(session, job.id)

    async def submit(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
        source_media: Sequence[ResolvedSourceMedia] = (),
    ) -> ProviderSubmitResult:
        """Delegate to GrokJobService.create_image_job or start_video_job."""
        if request.generation_type.output_kind is MediaKind.VIDEO:
            return await self._submit_video(
                request,
                user_id=user_id,
                session=session,
                billing_service=billing_service,
                account_id=account_id,
                token_cost=token_cost,
                product_id=product_id,
                source_job_id=source_job_id,
                primary_output_id=primary_output_id,
                primary_upload_id=primary_upload_id,
                source_media=source_media,
            )
        return await self._submit_image(
            request,
            user_id=user_id,
            session=session,
            billing_service=billing_service,
            account_id=account_id,
            token_cost=token_cost,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            primary_upload_id=primary_upload_id,
            source_media=source_media,
        )

    async def _resolve_source_urls(
        self,
        source_media: Sequence[ResolvedSourceMedia],
    ) -> list[str]:
        """Presign the already-resolved source assets for the Grok API."""
        urls: list[str] = []
        for source in source_media:
            url_result = await self._r2.get_presigned_url(
                source.storage_key,
                expires_in=_PRESIGN_EXPIRES_SECONDS,
            )
            urls.append(url_result.presigned_url)
        return urls

    async def _submit_image(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
        source_media: Sequence[ResolvedSourceMedia] = (),
    ) -> ProviderSubmitResult:
        """Delegate to GrokJobService.create_image_job."""
        urls = await self._resolve_source_urls(source_media)
        input_image_url = urls[0] if len(urls) == 1 else None
        input_image_urls = urls if len(urls) > 1 else None

        return await self._grok.create_image_job(
            session=session,
            user_id=user_id,
            prompt=request.prompt,
            model=request.model,
            generation_type=request.generation_type,
            n=request.n,
            aspect_ratio=request.aspect_ratio,
            name=request.name,
            negative_prompt=request.negative_prompt,
            input_image_url=input_image_url,
            input_image_urls=input_image_urls,
            billing_service=billing_service,
            account_id=account_id,
            token_cost=token_cost,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            primary_upload_id=primary_upload_id,
        )

    async def _submit_video(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
        product_id: str,
        source_job_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
        source_media: Sequence[ResolvedSourceMedia] = (),
    ) -> ProviderSubmitResult:
        """Delegate to GrokJobService.start_video_job."""
        urls = await self._resolve_source_urls(source_media)
        input_image_url = urls[0] if urls else None

        return await self._grok.start_video_job(
            session=session,
            user_id=user_id,
            prompt=request.prompt,
            model=request.model,
            generation_type=request.generation_type,
            duration=request.duration,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            name=request.name,
            input_image_url=input_image_url,
            input_video_url=request.input_video_url,
            billing_service=billing_service,
            account_id=account_id,
            token_cost=token_cost,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            primary_upload_id=primary_upload_id,
        )
