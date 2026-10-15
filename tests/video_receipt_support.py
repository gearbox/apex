"""Stubs for the accepted-task receipt collaborators of the video job services."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.api.services.video_jobs.protocol import OrphanCancelOutcome
from src.api.services.video_jobs.receipts import AcceptedTaskReceipts


def stub_receipts(*, recorded: bool = True) -> MagicMock:
    """An ``AcceptedTaskReceipts`` whose writes succeed (or report ``recorded``)."""
    receipts = MagicMock(spec=AcceptedTaskReceipts)
    receipts.record = AsyncMock(return_value=recorded)
    receipts.claim_for_request = AsyncMock()
    receipts.resolve = AsyncMock()
    return receipts


def stub_compensator(
    outcome: OrphanCancelOutcome = OrphanCancelOutcome.CANCELLED,
) -> MagicMock:
    """An ``OrphanTaskCompensator`` whose cancel returns ``outcome``."""
    compensator = MagicMock()
    compensator.cancel = AsyncMock(return_value=outcome)
    compensator.observe = AsyncMock()
    return compensator
