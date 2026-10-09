"""Errors raised by the Model Studio client.

Messages never carry response bodies, keys, URLs or prompts; the normalized
``ProviderFailure`` is the only structured payload.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.api.services.generation.provider_failures import ProviderFailure


class ModelStudioAPIError(Exception):
    """A classified Model Studio failure."""

    def __init__(self, failure: ProviderFailure) -> None:
        self.failure = failure
        super().__init__(f"Model Studio request failed ({failure.kind.value})")


class ModelStudioTransientError(ModelStudioAPIError):
    """A poll failure that must not fail the job: retry on the next tick."""
