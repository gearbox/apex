"""Shared bounds for reasons persisted on GPU sessions."""

from __future__ import annotations

MAX_GPU_SESSION_FAILURE_REASON_LENGTH = 500


def bounded_failure_reason(reason: str) -> str:
    """Keep a user-visible/persisted failure reason within its single shared bound."""
    return reason[:MAX_GPU_SESSION_FAILURE_REASON_LENGTH]
