"""OAuth login/signup request and response schemas (see docs/contracts/oauth-contract.md)."""

from __future__ import annotations

from typing import Annotated

import msgspec

# Runtime import: msgspec resolves struct annotations at runtime.
from src.api.services.legal.acceptance import AcceptedDocument

# Server-issued opaque values are 43 url-safe characters; the bounds only
# keep garbage (and oversized Redis key lookups) out.
_OpaqueValue = Annotated[
    str, msgspec.Meta(min_length=16, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")
]


class OAuthExchangeRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """Redeem the one-time login handoff code from the callback fragment."""

    code: _OpaqueValue


class OAuthSignupInfoRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """Look up a pending signup (non-consuming)."""

    ticket: _OpaqueValue


class OAuthCompleteSignupRequest(msgspec.Struct, kw_only=True, forbid_unknown_fields=True):
    """Finish an OAuth signup: legal acceptance (+ optional display name)."""

    ticket: _OpaqueValue
    accepted_documents: list[AcceptedDocument]
    """Exactly the product's required legal documents at their current
    versions (from ``GET /v1/legal/current``). ``[]`` when none are required."""
    display_name: Annotated[str, msgspec.Meta(min_length=1, max_length=100)] | None = None


class OAuthSignupInfoResponse(msgspec.Struct, kw_only=True):
    """What the signup screen shows about the pending identity."""

    email: str
    provider: str
