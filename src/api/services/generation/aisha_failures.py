"""Safe, stable public failures for Aisha/GPU generation paths.

The Aisha worker receives diagnostics from tunnels, ComfyUI, and GPU lifecycle
code. Those diagnostics are useful internally but are never a public API
contract. This enum is the only source of public failure codes/messages for
those paths.
"""

from __future__ import annotations

from enum import StrEnum


class AishaFailure(StrEnum):
    """Curated user-visible Aisha/GPU failure categories."""

    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_EXECUTION_FAILED = "provider_execution_failed"
    PROVIDER_TIMEOUT = "provider_timeout"
    GENERATION_SESSION_TERMINATED = "generation_session_terminated"

    @property
    def public_message(self) -> str:
        """Return the fixed safe message associated with this category."""
        return _PUBLIC_MESSAGES[self]


_PUBLIC_MESSAGES: dict[AishaFailure, str] = {
    AishaFailure.PROVIDER_UNAVAILABLE: "Generation infrastructure is temporarily unavailable.",
    AishaFailure.PROVIDER_EXECUTION_FAILED: "The generation engine could not complete the request.",
    AishaFailure.PROVIDER_TIMEOUT: (
        "Generation timed out before the compute service returned a result."
    ),
    AishaFailure.GENERATION_SESSION_TERMINATED: (
        "Generation stopped because the compute session ended."
    ),
}
