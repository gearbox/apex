"""Shared SSRF guard for Cloudflare tunnel hostnames.

Used by both AishaGenerationProvider.submit and AishaJobPoller._poll_one
to ensure per-session ComfyUIClient construction always validates the
tunnel hostname against the same allowlist semantics.

Two checks are required:
  1. Charset  — matches DNS-hostname pattern (lowercase, digits, hyphens, dots).
               Defeats embedded URL-meta characters (``?``, ``#``, ``@``, ``/``)
               that could smuggle a different host through urllib even when the
               endswith suffix check passes.
  2. Suffix   — ends with the configured tunnel domain (e.g. ``.gpu.cloudin.space``).
"""

from __future__ import annotations

import re
from typing import Final

_TUNNEL_HOST_RE: Final = re.compile(r"^[a-z0-9][a-z0-9-]*(\.[a-z0-9][a-z0-9-]*)+$")


class InvalidTunnelHostnameError(ValueError):
    """Raised when a tunnel hostname fails charset or suffix validation."""


def validate_tunnel_hostname(
    hostname: str,
    *,
    allowed_suffix: str,
) -> None:
    """Raise InvalidTunnelHostnameError if hostname fails any check.

    Args:
        hostname: The tunnel hostname to validate (e.g. ``abc123.gpu.cloudin.space``).
        allowed_suffix: The required hostname suffix including leading dot
            (e.g. ``".gpu.cloudin.space"``). Must be non-empty.

    Raises:
        InvalidTunnelHostnameError: If hostname is empty, fails the charset
            regex, contains ``..``, or does not end with ``allowed_suffix``.
    """
    if not hostname:
        raise InvalidTunnelHostnameError("Tunnel hostname is empty")

    if not allowed_suffix:
        raise InvalidTunnelHostnameError("allowed_suffix must be non-empty")

    if not _TUNNEL_HOST_RE.match(hostname):
        raise InvalidTunnelHostnameError(
            f"Tunnel hostname {hostname!r} contains invalid characters"
        )

    if ".." in hostname:
        raise InvalidTunnelHostnameError(f"Tunnel hostname {hostname!r} contains consecutive dots")

    if not hostname.endswith(allowed_suffix):
        raise InvalidTunnelHostnameError(
            f"Tunnel hostname {hostname!r} does not end with allowed suffix {allowed_suffix!r}"
        )
