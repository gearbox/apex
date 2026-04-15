"""Grok generation provider — adapter for GrokJobService."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.services.grok.job_service import GrokJobService
from src.api.services.storage import R2StorageService

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.db.models.storage import GenerationJob

logger = structlog.get_logger(__name__)


class GrokGenerationProvider:
    """Adapts GrokJobService to the GenerationProvider protocol.

    Model-generation_type compatibility and n-cap validation are handled
    by the GenerationService orchestrator via ModelType enum properties.
    This class only contains Grok-specific submission logic.
    """

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
        source_output_id: UUID | None = None,
    ) -> GenerationJob:
        """Delegate to GrokJobService.create_image_job or start_video_job."""
        if request.generation_type.is_video:
            return await self._submit_video(
                request,
                user_id=user_id,
                session=session,
                billing_service=billing_service,
                account_id=account_id,
                token_cost=token_cost,
                product_id=product_id,
                source_job_id=source_job_id,
                source_output_id=source_output_id,
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
            source_output_id=source_output_id,
        )

    async def _resolve_input_url(
        self,
        request: UnifiedGenerationRequest,
        session: AsyncSession,
    ) -> str | None:
        """Resolve input URL from either source_output_id or input_image_id."""
        if request.source_output_id is not None:
            from src.db.repositories.output import OutputRepository

            output = await OutputRepository(session).get(request.source_output_id)
            if output is None:
                raise ValueError(f"Source output {request.source_output_id} not found")
            url_result = await self._r2.get_presigned_url(output.storage_key, expires_in=3600)
            return url_result.presigned_url

        if request.input_image_id is None:
            return None

        from src.db.repositories.user_image import UserImageRepository

        input_image = await UserImageRepository(session).get(request.input_image_id)
        if input_image is None:
            raise ValueError(f"Input image {request.input_image_id} not found")
        url_result = await self._r2.get_presigned_url(
            input_image.storage_key,
            expires_in=3600,
        )
        return url_result.presigned_url

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
        source_output_id: UUID | None = None,
    ) -> GenerationJob:
        """Delegate to GrokJobService.create_image_job."""
        input_image_url = await self._resolve_input_url(request, session)

        job = await self._grok.create_image_job(
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
            input_image_id=request.input_image_id,
            billing_service=billing_service,
            account_id=account_id,
            token_cost=token_cost,
            product_id=product_id,
            source_job_id=source_job_id,
            source_output_id=source_output_id,
        )
        if job is None:
            raise ValueError("Grok image job creation returned None")
        return job

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
        source_output_id: UUID | None = None,
    ) -> GenerationJob:
        """Delegate to GrokJobService.start_video_job."""
        input_image_url = await self._resolve_input_url(request, session)

        job = await self._grok.start_video_job(
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
            source_output_id=source_output_id,
        )
        if job is None:
            raise ValueError("Grok video job creation returned None")
        return job
