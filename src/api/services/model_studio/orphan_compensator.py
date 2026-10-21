"""Cancel or observe a Model Studio task that no job row references."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.api.services.model_studio.errors import ModelStudioAPIError, ModelStudioTransientError
from src.api.services.model_studio.types import ModelStudioCancelResult, ModelStudioTaskStatus
from src.api.services.video_jobs.protocol import (
    OrphanCancelOutcome,
    OrphanObservation,
    OrphanTaskState,
)
from src.core.enums import Provider

if TYPE_CHECKING:
    from src.api.services.model_studio.client import ModelStudioClient

_NOT_FOUND_STATUS = 404


class ModelStudioOrphanCompensator:
    """``OrphanTaskCompensator`` for Wan (Model Studio)."""

    provider: ClassVar[Provider] = Provider.ALIBABA
    transient_errors: ClassVar[tuple[type[Exception], ...]] = (ModelStudioTransientError,)
    permanent_errors: ClassVar[tuple[type[Exception], ...]] = (ModelStudioAPIError,)

    def __init__(self, client: ModelStudioClient) -> None:
        self._client = client

    async def cancel(self, external_task_id: str) -> OrphanCancelOutcome:
        """Cancel the task; upstream accepts this only while it is ``PENDING``."""
        result = await self._client.cancel_task(external_task_id)
        if result is ModelStudioCancelResult.CANCELLED:
            return OrphanCancelOutcome.CANCELLED
        return OrphanCancelOutcome.NOT_CANCELLABLE

    async def observe(self, external_task_id: str) -> OrphanObservation:
        """Read the task status; an unknown/expired task is ``GONE``."""
        try:
            task = await self._client.get_task(external_task_id)
        except ModelStudioTransientError:
            raise
        except ModelStudioAPIError as exc:
            if exc.failure.provider_status_code == _NOT_FOUND_STATUS:
                return OrphanObservation(state=OrphanTaskState.GONE, upstream_status=None)
            raise
        status = task.task_status
        if status is ModelStudioTaskStatus.UNKNOWN:
            return OrphanObservation(state=OrphanTaskState.GONE, upstream_status=None)
        if status is ModelStudioTaskStatus.PENDING:
            return OrphanObservation(state=OrphanTaskState.PENDING, upstream_status=status.value)
        if status is ModelStudioTaskStatus.CANCELED:
            return OrphanObservation(state=OrphanTaskState.CANCELLED, upstream_status=status.value)
        return OrphanObservation(state=OrphanTaskState.OTHER, upstream_status=status.value)
