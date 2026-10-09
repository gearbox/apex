"""Wan generation provider — adapter for WanJobService (Alibaba Model Studio)."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.api.services.generation.generation_modes import resolve_generation_modes
from src.api.services.generation.video_validation import input_video_seconds
from src.api.services.model_studio.wan_payload import build_wan_video_payload
from src.core.enums import GenerationType, JobStatus, MediaKind
from src.core.model_registry import get_model_meta

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.api.services.generation.base import ProviderSubmitResult
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.api.services.model_studio.wan_job_service import WanJobService
    from src.api.services.storage import R2StorageService
    from src.db.models.storage import GenerationJob


class WanGenerationProvider:
    """Adapts ``WanJobService`` to the ``GenerationProvider`` protocol.

    Every rule about duration, resolution, ratio, media cardinality and optional
    parameters lives in the registry and the orchestrator; this class only
    translates an already-validated request into a Model Studio task.
    """

    # Wan outputs are video only; request-level image sizing does not apply.
    supports_image_sizing: ClassVar[bool] = False

    def __init__(
        self,
        wan_job_service: WanJobService,
        r2_storage: R2StorageService,
        *,
        input_url_ttl_seconds: int,
    ) -> None:
        self._wan = wan_job_service
        self._r2 = r2_storage
        self._input_url_ttl_seconds = input_url_ttl_seconds

    def validate(self, request: UnifiedGenerationRequest) -> None:
        """No-op: all Wan rules are registry-driven and enforced by the orchestrator."""

    async def refresh_job(
        self,
        session: AsyncSession,
        job: GenerationJob,
    ) -> GenerationJob | None:
        """Poll Model Studio for a fresh status on queued/running jobs."""
        if GenerationType(job.generation_type).output_kind is not MediaKind.VIDEO:
            return None
        if job.status not in (JobStatus.QUEUED.value, JobStatus.RUNNING.value):
            return None
        return await self._wan.poll_video_job(session, job.id)

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
        """Presign the inputs, build the task body and start the job."""
        video = get_model_meta(request.model).video
        if video is None:
            raise ValueError(f"Model '{request.model.value}' has no video constraints")

        # Presigned URLs leave our perimeter: short-lived, never logged.
        media: list[tuple[MediaKind, str]] = []
        for source in source_media:
            url_result = await self._r2.get_presigned_url(
                source.storage_key, expires_in=self._input_url_ttl_seconds
            )
            media.append((source.media_kind, url_result.presigned_url))

        contract = resolve_generation_modes(request.model, capabilities=None)[
            request.generation_type
        ].source_media
        payload = build_wan_video_payload(
            model=request.model,
            prompt=request.prompt,
            contract=contract,
            media=media,
            duration=request.duration,
            resolution=request.resolution,
            aspect_ratio=request.aspect_ratio,
            audio=request.audio,
            prompt_extend=request.prompt_extend,
            seed=request.seed,
        )
        return await self._wan.start_video_job(
            session,
            user_id=user_id,
            prompt=request.prompt,
            model=request.model,
            generation_type=request.generation_type,
            payload=payload,
            aspect_ratio=request.aspect_ratio,
            name=request.name,
            billing_service=billing_service,
            account_id=account_id,
            token_cost=token_cost,
            billed_seconds=request.duration + input_video_seconds(source_media, video),
            video_resolution=request.resolution,
            product_id=product_id,
            source_job_id=source_job_id,
            primary_output_id=primary_output_id,
            primary_upload_id=primary_upload_id,
        )
