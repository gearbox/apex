"""Contract between an async-video provider service and ``AsyncVideoJobWorker``."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Protocol

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.video_jobs.outcome import VideoPollOutcome
    from src.core.enums import Provider
    from src.db.models import GenerationJob


class AsyncVideoJobService(Protocol):
    """One provider's polling and settlement entry points."""

    provider: ClassVar[Provider]
    transient_poll_errors: ClassVar[tuple[type[Exception], ...]]
    """Poll errors that mean "try again next tick" rather than "the job failed".

    Each must carry a ``failure: ProviderFailure`` attribute for logging.
    """
    running_progress_pct: ClassVar[int | None]
    """A legacy fabricated percentage some providers still emit. New providers MUST use None."""

    async def poll_video_job_for_worker(
        self, session: AsyncSession, job_id: UUID
    ) -> VideoPollOutcome:
        """Poll one job without mutating its status."""
        ...

    async def settle_video_poll_outcome(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        outcome: VideoPollOutcome,
        product_id: str,
    ) -> GenerationJob:
        """Apply a poll outcome exactly once through the shared transitions."""
        ...
