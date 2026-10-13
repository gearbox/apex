"""Apply one poll outcome through the shared job-state transitions."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.api.services.generation.provider_failures import ProviderFailure
from src.core.enums import VideoPollStatus

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.job_state_transition import JobStateTransitionService
    from src.api.services.video_jobs.materializer import RemoteVideoMaterializer
    from src.api.services.video_jobs.outcome import VideoPollOutcome
    from src.db.models import GenerationJob


async def settle_video_poll_outcome(
    session: AsyncSession,
    *,
    job_id: UUID,
    outcome: VideoPollOutcome,
    product_id: str,
    transition_service: JobStateTransitionService,
    materializer: RemoteVideoMaterializer,
    settle_failure: Callable[[AsyncSession, UUID, VideoPollOutcome, str], Awaitable[GenerationJob]],
) -> GenerationJob:
    """Apply a video poll result exactly once.

    Read-through and worker callers both end here. Conditional state updates in
    ``JobStateTransitionService`` make repeated reads and worker/read races
    no-ops after the first terminal settlement, including its refund and
    terminal event.

    Args:
        session: Database session.
        job_id: The job being settled.
        outcome: What the provider reported.
        product_id: Product scope.
        transition_service: Settlement primitive for this session.
        materializer: Completes a finished upstream video.
        settle_failure: The owning service's own settle entry point, used by the
            materializer for an undeliverable output.
    """
    if outcome.status == VideoPollStatus.STILL_RUNNING:
        return await transition_service.transition_to_running(job_id)
    if outcome.status == VideoPollStatus.COMPLETED:
        if outcome.result is None:
            return await transition_service.transition_to_completed(
                job_id, outputs=[], product_id=product_id
            )
        return await materializer.finalize_completed_video(
            session,
            job_id=job_id,
            result=outcome.result,
            product_id=product_id,
            settle_failure=settle_failure,
        )

    failure = outcome.failure
    job, _ = await transition_service.transition_to_failed(
        job_id,
        public_error_message=(
            ProviderFailure.public_message_for_failure(failure)
            if failure is not None
            else "The AI provider could not complete the request."
        ),
        failure_code=failure.public_code if failure is not None else None,
        refund=not failure.billable if failure is not None else True,
        product_id=product_id,
    )
    return job
