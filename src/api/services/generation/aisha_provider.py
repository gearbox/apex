"""Aisha generation-provider facade and media-handler registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.api.services.bundle_index import BundleIndexService, BundleNotFoundError
from src.api.services.generation.aisha.handlers import (
    AishaImageGenerationHandler,
    AishaMediaHandler,
    UnsupportedMediaError,
)
from src.core.enums import MediaKind, ModelType

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.api.services.generation.base import ProviderSubmitResult
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.api.services.gpu_session.service import GpuSessionService
    from src.api.services.storage import R2StorageService
    from src.api.services.workflow import WorkflowService
    from src.api.services.workflow.contract import BundleCapabilities
    from src.db.models.storage import GenerationJob


class AishaGenerationProvider:
    """Route Aisha requests to a handler selected by output media kind."""

    supports_image_sizing: ClassVar[bool] = True

    def __init__(
        self,
        workflow_service: WorkflowService,
        gpu_session_service: GpuSessionService | None,
        bundle_index: BundleIndexService,
        r2_storage: R2StorageService | None = None,
        tunnel_domain: str = "",
        tunnel_hostname_allowed_prefix: str | None = None,
        max_input_megapixels: float = 100.0,
        handlers: Mapping[MediaKind, AishaMediaHandler] | None = None,
    ) -> None:
        self._bundle_index = bundle_index
        self._handlers = (
            dict(handlers)
            if handlers is not None
            else {
                MediaKind.IMAGE: AishaImageGenerationHandler(
                    workflow_service=workflow_service,
                    gpu_session_service=gpu_session_service,
                    bundle_index=bundle_index,
                    r2_storage=r2_storage,
                    tunnel_domain=tunnel_domain,
                    tunnel_hostname_allowed_prefix=tunnel_hostname_allowed_prefix,
                    max_input_megapixels=max_input_megapixels,
                )
            }
        )

    def _capabilities_for(self, model: ModelType) -> BundleCapabilities | None:
        """Read the indexed capability used for media-handler validation."""
        try:
            mapping = self._bundle_index.resolve_bundle(model.value)
        except BundleNotFoundError:
            return None
        return mapping.capabilities

    def _handler_for(self, media: MediaKind) -> AishaMediaHandler:
        """Select a registered output-media handler without model branching."""
        try:
            return self._handlers[media]
        except KeyError as exc:
            raise UnsupportedMediaError(
                f"Aisha {media.value} media has no registered handler"
            ) from exc

    def validate(self, request: UnifiedGenerationRequest) -> None:
        """Run validation owned by the handler for the requested output medium."""
        self._handler_for(request.generation_type.output_kind).validate(
            request, self._capabilities_for(request.model)
        )

    async def refresh_job(
        self,
        session: AsyncSession,  # noqa: ARG002
        job: GenerationJob,  # noqa: ARG002
    ) -> GenerationJob | None:
        """No-op — Aisha job status is updated by the background AishaJobPoller."""
        return None

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
        """Submit through the handler registered for the request output medium."""
        handler = self._handler_for(request.generation_type.output_kind)
        return await handler.submit(
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
