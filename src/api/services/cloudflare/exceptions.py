"""Exceptions for Cloudflare API errors."""

from __future__ import annotations


class CloudflareError(Exception):
    """Base exception for Cloudflare API errors."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TunnelCreationError(CloudflareError):
    """Raised when tunnel creation fails."""


class TunnelConfigError(CloudflareError):
    """Raised when tunnel configuration fails."""


class DNSRecordError(CloudflareError):
    """Raised when DNS record creation fails."""


class TunnelDeletionError(CloudflareError):
    """Raised when tunnel deletion fails."""
