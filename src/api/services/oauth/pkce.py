"""Random flow values, PKCE (RFC 7636 S256), and ``return_to`` validation."""

from __future__ import annotations

import base64
import hashlib
import re
import secrets
from typing import Final

# 48 random bytes → 64 url-safe characters, inside RFC 7636's 43-128 range.
_VERIFIER_BYTES: Final = 48
# 32 random bytes → 43 url-safe characters for state / nonce / binding / codes.
_TOKEN_BYTES: Final = 32

RETURN_TO_MAX_LENGTH: Final = 512
# A same-origin absolute path: starts with one "/", not "//" or "/\" (both
# are protocol-relative in browsers), no whitespace, control chars, or "#".
_RETURN_TO_RE: Final = re.compile(r"/(?![/\\])[^\s#\x00-\x1f\x7f]*")


def generate_verifier() -> str:
    """A fresh PKCE code_verifier (64 url-safe characters)."""
    return secrets.token_urlsafe(_VERIFIER_BYTES)


def generate_opaque_token() -> str:
    """A fresh unguessable value for state, nonce, binding, handoff code, or ticket."""
    return secrets.token_urlsafe(_TOKEN_BYTES)


def s256_challenge(verifier: str) -> str:
    """``BASE64URL(SHA256(verifier))`` with padding stripped (RFC 7636 §4.2)."""
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def validate_return_to(value: str | None) -> str | None:
    """Accept ``None`` or a same-origin path; reject anything that could leave the app.

    Uses ``fullmatch`` — ``re.match(...$)`` would accept a trailing newline.

    Raises:
        ValueError: Absolute/protocol-relative URL, whitespace, fragment, or too long.
    """
    if value is None:
        return None
    if len(value) > RETURN_TO_MAX_LENGTH or _RETURN_TO_RE.fullmatch(value) is None:
        raise ValueError("return_to must be a same-origin path")
    return value
