"""Generation provider protocol — Strategy pattern interface."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BalanceEvent, BillingService
    from src.api.services.generation.source_media import ResolvedSourceMedia
    from src.db.models.storage import GenerationJob


@dataclasses.dataclass(frozen=True, slots=True)
class ProviderSubmitResult:
    """Result of ``GenerationProvider.submit()``.

    ``balance_event`` is the pending SSE event for the ``check_and_reserve``
    debit performed inside ``submit()``, threaded back to the orchestrator so
    it can publish strictly after its own post-submit commit (never before —
    see ``EventBus.publish_balance``).

    ``balance_after`` is the post-debit balance from the reserve transaction
    (``BillingResult.txn.balance_after``), for the API response — distinct
    from ``balance_event``, which is the optional SSE payload and must not be
    used as the response balance source (it's ``None`` whenever there's no
    SSE target).
    """

    job: GenerationJob
    balance_after: int
    balance_event: BalanceEvent | None = None


class GenerationProvider(Protocol):
    """Protocol for provider-specific generation strategies.

    Each provider implements validation and submission.
    Billing is handled by the orchestrator (GenerationService), NOT the provider.
    """

    # Whether this provider honours request-level image sizing fields
    # (image_resolution / width / height). Grok ignores them (fixed output
    # sizes); Aisha applies them via resolve_dimensions().
    supports_image_sizing: ClassVar[bool] = True

    def validate(self, request: UnifiedGenerationRequest) -> None:
        """Validate provider-specific constraints.

        The base model-generation_type compatibility check is done by the
        orchestrator via ``ModelType.supports_generation_type()``. This method
        handles provider-specific constraints only (e.g., provider-specific
        aspect_ratio restrictions, prompt length limits, etc.).

        Raises:
            ValueError: If the request is invalid for this provider.
        """
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
        """Submit the generation request to the provider.

        The billing saga (check_and_reserve) is called by the GenerationService
        orchestrator BEFORE this method. The provider receives the pre-reserved
        account_id and token_cost to stamp on the job record.

        Args:
            request: Validated unified request.
            user_id: Authenticated user.
            session: DB session (for job creation and commit).
            billing_service: For stamping the debit transaction on the job.
            account_id: Pre-resolved token account.
            token_cost: Pre-calculated cost.
            product_id: Product this generation belongs to.
            source_media: Ordered, ownership-checked source assets. Providers
                use their storage data directly and never re-resolve request
                aliases.

        Returns:
            ``ProviderSubmitResult`` wrapping the created GenerationJob (status
            may be QUEUED, COMPLETED, or RUNNING depending on whether the
            provider is sync or async) and the pending balance event, if any.

        Raises:
            GenerationError: On provider failure (triggers refund in orchestrator).
        """
        ...

    async def refresh_job(
        self,
        session: AsyncSession,
        job: GenerationJob,
    ) -> GenerationJob | None:
        """Optional poll-on-read hook for providers with async job status.

        Default: no-op (returns ``None`` — caller keeps serving the DB row
        as-is). Providers with a polling backend (e.g. Grok video) override
        this to fetch fresh status; the orchestrator is responsible for
        catching poll failures and serving the stale row.
        """
        ...
