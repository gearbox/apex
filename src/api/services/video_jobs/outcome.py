"""Value objects exchanged between async-video job services and their worker."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.services.generation.provider_failures import ProviderFailure
    from src.core.enums import VideoPollStatus


@dataclasses.dataclass(frozen=True, slots=True)
class RemoteVideoResult:
    """A completed upstream video, reduced to what materialization needs."""

    url: str
    provider_request_id: str | None = None


@dataclasses.dataclass(frozen=True)
class VideoPollOutcome:
    """Result of polling a provider for one video job, independent of JobStatus.

    ``poll_video_job_for_worker`` returns this instead of mutating job.status
    directly — the periodic worker drives the actual state transition (and
    any refund) through JobStateTransitionService.
    """

    status: VideoPollStatus
    error_message: str | None = None
    failure: ProviderFailure | None = None
    result: RemoteVideoResult | None = None
