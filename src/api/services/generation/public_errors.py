"""Public-safe failure-message helpers for generation jobs.

``GenerationJob.error_message`` predates the public API contract and may
contain internal diagnostics.  It must never be serialized to an end user.
"""

from __future__ import annotations

from src.core.enums import JobStatus

LEGACY_FAILED_JOB_MESSAGE = "Generation failed. Please try again."


def public_error_for_job(*, status: str, public_error_message: str | None) -> str | None:
    """Return the only failure text that may cross the user-facing boundary.

    Older failed rows do not have ``public_error_message``.  A fixed fallback
    keeps those rows useful without ever disclosing their legacy diagnostic
    ``error_message`` value.
    """
    if public_error_message:
        return public_error_message
    return LEGACY_FAILED_JOB_MESSAGE if status == JobStatus.FAILED.value else None
