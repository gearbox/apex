"""Contract between an async-video provider service and ``AsyncVideoJobWorker``."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
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
    VIDEO_POLLER_NAME: ClassVar[str]
    """Worker (and leader-lease) name. The single definition per provider: the in-process
    worker and the standalone process must agree on it to share one lease, and it must stay
    byte-for-byte stable across releases."""
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


class OrphanCancelOutcome(StrEnum):
    """Result of asking a provider to cancel an orphaned task."""

    CANCELLED = "cancelled"
    NOT_CANCELLABLE = "not_cancellable"
    """The provider refused: the task is already past the cancellable state."""
    UNSUPPORTED = "unsupported"
    """The provider has no cancel operation."""


class OrphanTaskState(StrEnum):
    """What the reconciler needs to know about an upstream task."""

    PENDING = "pending"
    CANCELLED = "cancelled"
    OTHER = "other"
    """Running or finished (succeeded / failed): not cancellable any more."""
    GONE = "gone"
    """The provider does not know the task (never existed or expired)."""


@dataclass(frozen=True, slots=True)
class OrphanObservation:
    """One provider status read for an orphan."""

    state: OrphanTaskState
    upstream_status: str | None
    """The provider's own status string, verbatim; ``None`` when ``state`` is GONE."""


class OrphanTaskCompensator(Protocol):
    """Cancel or observe an upstream task that no job row references.

    Both methods raise one of ``transient_errors`` when the provider could not be
    reached or throttled the call, so the next tick retries; neither ever returns
    a fabricated value. Any other exception is a permanent failure to compensate.
    """

    provider: ClassVar[Provider]
    transient_errors: ClassVar[tuple[type[Exception], ...]]

    async def cancel(self, external_task_id: str) -> OrphanCancelOutcome:
        """Ask the provider to cancel the task."""
        ...

    async def observe(self, external_task_id: str) -> OrphanObservation:
        """Read the task's status without fetching any output."""
        ...
