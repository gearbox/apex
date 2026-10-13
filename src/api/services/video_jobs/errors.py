"""Errors raised by the shared async-video completion path."""

from __future__ import annotations


class ProviderOutputUnavailableError(Exception):
    """The provider's result URL is permanently unavailable (an HTTP 4xx that will not heal).

    Raised before any materialization attempt exists. The message carries the status
    only — the result URL is a presigned credential and must never be logged.
    """

    def __init__(self, status_code: int) -> None:
        super().__init__(f"provider result URL is not available (HTTP {status_code})")
        self.status_code = status_code
