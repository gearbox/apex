"""Generation provider protocol — Strategy pattern interface."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.db.models.storage import GenerationJob


class GenerationProvider(Protocol):
    """Protocol for provider-specific generation strategies.

    Each provider implements validation and submission.
    Billing is handled by the orchestrator (GenerationService), NOT the provider.
    """

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
        source_output_id: UUID | None = None,
    ) -> GenerationJob:
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

        Returns:
            The created GenerationJob (status may be QUEUED, COMPLETED, or RUNNING
            depending on whether the provider is sync or async).

        Raises:
            GenerationError: On provider failure (triggers refund in orchestrator).
        """
        ...
