"""The media-handler seam used by the Aisha generation-provider facade."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

from src.api.services.generation.service import FeatureNotSupportedError
from src.core.enums import MediaKind

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.api.services.generation.base import ProviderSubmitResult
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.api.services.workflow.contract import BundleCapabilities, MediaSlot


class UnsupportedMediaError(FeatureNotSupportedError):
    """A bundle output kind has no registered Aisha submit handler."""


class AishaMediaHandler(Protocol):
    """Per-output-media submit strategy owned by the Aisha facade."""

    media: ClassVar[MediaKind]

    def validate(
        self,
        request: UnifiedGenerationRequest,
        capabilities: BundleCapabilities | None,
    ) -> None:
        """Validate media-specific request semantics."""
        ...

    async def prepare_inputs(
        self,
        source_media: Sequence[ResolvedSourceMedia],
    ) -> dict[MediaSlot, list[str]]:
        """Prepare uploaded ComfyUI filenames keyed by their declared slot."""
        ...

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
        """Submit a request for this output medium."""
        ...


class AishaImageGenerationHandler:
    """Image submit handler delegating to the established ComfyUI image flow.

    Keeping the callable injected is deliberate: the facade owns session and
    handler selection, while the image flow retains its thoroughly tested
    upload, dimensions, and queueing implementation during this extraction.
    """

    media: ClassVar[MediaKind] = MediaKind.IMAGE

    def __init__(
        self,
        submit_image: Callable[..., Awaitable[ProviderSubmitResult]],
    ) -> None:
        self._submit_image = submit_image

    def validate(
        self,
        request: UnifiedGenerationRequest,
        capabilities: BundleCapabilities | None,
    ) -> None:
        """Image flow has no additional validation beyond its bound bundle."""

    async def prepare_inputs(
        self,
        source_media: Sequence[ResolvedSourceMedia],  # noqa: ARG002
    ) -> dict[MediaSlot, list[str]]:
        """The image submit flow prepares uploads after opening the session tunnel."""
        return {}

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
        """Delegate to the image implementation after registry selection."""
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
