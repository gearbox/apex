"""The ``apex_oauth_tx`` browser-binding cookie for the OAuth flow.

``/authorize`` mints a random binding value, stores only its SHA-256 in the
Redis flow state, and sets the raw value in this cookie. Every later step
(callback, exchange, signup-info, complete-signup) requires the same cookie,
so a handoff code or signup ticket leaked or injected into another browser
is useless there (login-CSRF / session-swap defence).

Host-only (no Domain) on the API host and scoped to ``/v1/auth/oauth``.
``SameSite=Lax`` works because the provider callback is a top-level GET
navigation; a ``response_mode=form_post`` provider (Apple) would need
``SameSite=None`` — deliberately left to that provider's arc.
"""

from __future__ import annotations

import hashlib
from typing import Final

from litestar.datastructures import Cookie

OAUTH_TX_COOKIE: Final = "apex_oauth_tx"
OAUTH_TX_COOKIE_PATH: Final = "/v1/auth/oauth"


def binding_hash(binding: str) -> str:
    """SHA-256 hex digest of a binding value (what Redis stores — never the raw value)."""
    return hashlib.sha256(binding.encode()).hexdigest()


def mint_oauth_tx_cookie(binding: str, *, max_age: int, secure: bool) -> Cookie:
    """Set-Cookie carrying the raw binding value.

    Args:
        binding: Random per-flow binding value.
        max_age: Lifetime in seconds — ``oauth_signup_ticket_ttl_seconds`` so the
            cookie outlives the whole flow up to complete-signup.
        secure: ``Settings.content_cookie_secure`` (dropped only in dev).
    """
    return Cookie(
        key=OAUTH_TX_COOKIE,
        value=binding,
        httponly=True,
        secure=secure,
        samesite="lax",
        path=OAUTH_TX_COOKIE_PATH,
        domain=None,
        max_age=max_age,
    )


def clear_oauth_tx_cookie(*, secure: bool) -> Cookie:
    """Set-Cookie that expires the binding cookie (after exchange / complete-signup)."""
    return Cookie(
        key=OAUTH_TX_COOKIE,
        value="",
        httponly=True,
        secure=secure,
        samesite="lax",
        path=OAUTH_TX_COOKIE_PATH,
        domain=None,
        max_age=0,
    )
