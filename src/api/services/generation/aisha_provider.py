"""Aisha generation provider — adapter for ComfyUI WorkflowService."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

import structlog

from src.api.schemas.generation import DEFAULT_NEGATIVE_PROMPT, GenerationRequest
from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.workflow_service import WorkflowService
from src.core.enums import JobStatus, Provider
from src.core.uid import new_id
from src.db import StorageRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.schemas.unified_generation import UnifiedGenerationRequest
    from src.api.services.billing import BillingService
    from src.db.models.storage import GenerationJob

logger = structlog.get_logger(__name__)


class AishaGenerationProvider:
    """Adapts the ComfyUI stack (WorkflowService) to the GenerationProvider protocol.

    Model-generation_type compatibility and n-cap validation are handled
    by the GenerationService orchestrator via ModelType enum properties.
    This class only contains Aisha/ComfyUI-specific submission logic.

    Note: Aisha I2I via the unified endpoint uses input_image_id (pre-uploaded to R2).
    The provider would need to download from R2 and re-upload to ComfyUI for I2I.
    TODO: Implement R2 -> ComfyUI image bridge for Aisha I2I via unified endpoint.
    """

    def __init__(
        self,
        comfyui_client: ComfyUIClient,
        workflow_service: WorkflowService,
    ) -> None:
        self._client = comfyui_client
        self._workflow = workflow_service

    def validate(self, request: UnifiedGenerationRequest) -> None:
        """Aisha-specific validation beyond what the enum provides."""
        from src.core.enums import ModelType

        if request.model == ModelType.AISHA_VIDEO:
            raise ValueError("Aisha video generation is not yet available via the unified endpoint")

    async def submit(
        self,
        request: UnifiedGenerationRequest,
        *,
        user_id: UUID,
        session: AsyncSession,
        billing_service: BillingService,
        account_id: UUID,
        token_cost: int,
    ) -> GenerationJob:
        """Build workflow, queue with ComfyUI, create DB job record.

        The full job lifecycle (polling, output storage, status transitions)
        is handled DB-side via AishaJobService on subsequent poll-on-read calls.
        """
        job_id = new_id()

        # Map unified request -> legacy GenerationRequest for workflow_service
        legacy_request = GenerationRequest(
            prompt=request.prompt,
            name=request.name,
            negative_prompt=request.negative_prompt or DEFAULT_NEGATIVE_PROMPT,
            height=request.height or 1024,
            aspect_ratio=request.aspect_ratio,
            model_type=request.model,
            generation_type=request.generation_type,
            max_images=request.n,
            seed=request.seed,
            steps=request.steps or 12,
        )

        # Create DB job record
        repo = StorageRepository(session)
        db_job = await repo.create_job(
            id=job_id,
            user_id=user_id,
            name=legacy_request.name or "Untitled",
            prompt=request.prompt,
            generation_type=request.generation_type,
            status=JobStatus.PENDING,
            provider=Provider.AISHA,
            model=request.model.value,
            aspect_ratio=request.aspect_ratio.value,
        )

        # Billing reservation
        txn = await billing_service.check_and_reserve(
            account_id,
            token_cost,
            job_id,
            metadata={
                "provider": Provider.AISHA.value,
                "generation_type": request.generation_type.value,
                "model": request.model.value,
            },
            session=session,
        )
        if db_job is not None:
            db_job.token_cost = token_cost
            db_job.debit_transaction_id = txn.id
        await session.flush()

        # Build and queue workflow
        workflow = self._workflow.load_workflow(request.model)
        self._workflow.validate_workflow(workflow)
        configured = self._workflow.apply_parameters(
            workflow=workflow,
            request=legacy_request,
            filename_prefix=f"gen_{str(job_id)[:8]}",
        )

        result = await self._client.queue_prompt(configured)
        if prompt_id := result.get("prompt_id"):
            if db_job is not None:
                db_job.status = JobStatus.QUEUED
                db_job.started_at = datetime.now(UTC)
                db_job.external_request_id = prompt_id
        elif db_job is not None:
            db_job.status = JobStatus.FAILED
            db_job.error_message = "No prompt_id from ComfyUI"

        if db_job is None:
            raise ValueError("Failed to create Aisha job record")

        await session.flush()
        return db_job
