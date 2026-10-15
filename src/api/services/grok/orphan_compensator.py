"""Observe an xAI video request that no job row references.

xAI documents no cancel or delete for deferred video requests
(https://docs.x.ai/docs/guides/video-generation describes only start and poll),
so ``cancel`` is ``UNSUPPORTED`` and the reconciler records such orphans as
``absorbed`` with their upstream status and an ops alert carrying the request id.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar

from src.api.services.grok import (
    GrokAPIError,
    GrokConnectionError,
    GrokDeferredExpiredError,
    GrokDeferredTerminalError,
    GrokInvalidRequestError,
    GrokModerationError,
    GrokRateLimitError,
    GrokTimeoutError,
)
from src.api.services.video_jobs.protocol import (
    OrphanCancelOutcome,
    OrphanObservation,
    OrphanTaskState,
)
from src.core.enums import Provider

if TYPE_CHECKING:
    from src.api.services.grok import GrokClient


class GrokOrphanCompensator:
    """``OrphanTaskCompensator`` for Grok video."""

    provider: ClassVar[Provider] = Provider.GROK
    transient_errors: ClassVar[tuple[type[Exception], ...]] = (
        GrokRateLimitError,
        GrokTimeoutError,
        GrokConnectionError,
        # An unclassified API error (e.g. upstream 5xx). Terminal, expired and
        # invalid-id answers are mapped inside ``observe`` and never reach here.
        GrokAPIError,
    )

    def __init__(self, grok_client: GrokClient) -> None:
        self._grok = grok_client

    async def cancel(self, external_task_id: str) -> OrphanCancelOutcome:  # noqa: ARG002
        """xAI has no documented cancel for video requests."""
        return OrphanCancelOutcome.UNSUPPORTED

    async def observe(self, external_task_id: str) -> OrphanObservation:
        """Read the deferred request status; the video itself is never fetched."""
        try:
            result = await self._grok.get_video_result(external_task_id)
        except (GrokModerationError, GrokDeferredTerminalError):
            return OrphanObservation(state=OrphanTaskState.OTHER, upstream_status="failed")
        except (GrokDeferredExpiredError, GrokInvalidRequestError):
            # Expired or unknown request id.
            return OrphanObservation(state=OrphanTaskState.GONE, upstream_status=None)
        if result is None:
            return OrphanObservation(state=OrphanTaskState.PENDING, upstream_status="pending")
        return OrphanObservation(state=OrphanTaskState.OTHER, upstream_status="done")
