"""Realistic GitHub token fixtures shared by the redaction unit and integration tests."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

_STATELESS_HEADER = {"alg": "RS256", "typ": "JWT"}
_STATELESS_CLAIMS = {
    "installation_id": 98765432,
    "iat": 1_793_000_000,
    "exp": 1_793_003_600,
    "permissions": "contents:read",
}
# RS256 over a 2048-bit key: a 256-byte signature, i.e. 342 base64url characters.
_STATELESS_SIGNATURE_BYTES = 256


def _b64url(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _json_b64url(value: Mapping[str, object]) -> str:
    return _b64url(json.dumps(value, separators=(",", ":")).encode())


def _signature(seed: str) -> bytes:
    """Deterministic, random-looking bytes — reproducible across runs, unlike ``os.urandom``."""
    out = b""
    counter = 0
    while len(out) < _STATELESS_SIGNATURE_BYTES:
        out += hashlib.sha256(f"{seed}:{counter}".encode()).digest()
        counter += 1
    return out[:_STATELESS_SIGNATURE_BYTES]


def stateless_ghs_jwt(app_id: int = 1234567) -> str:
    """The ``<hdr>.<payload>.<sig>`` part of a stateless installation token."""
    return ".".join(
        (
            _json_b64url(_STATELESS_HEADER),
            _json_b64url({**_STATELESS_CLAIMS, "app_id": app_id}),
            _b64url(_signature(f"stateless-ghs:{app_id}")),
        )
    )


def stateless_ghs(app_id: int = 1234567) -> str:
    """A realistic stateless installation token: ``ghs_<APPID>_<hdr>.<payload>.<sig>``.

    About 520 characters, base64url segments (``A-Za-z0-9-_``) joined by two dots.
    """
    return f"ghs_{app_id}_{stateless_ghs_jwt(app_id)}"
