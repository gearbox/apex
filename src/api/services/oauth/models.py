"""Values persisted between OAuth steps (Redis, msgspec-JSON) and callback outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

import msgspec

# Runtime import required: msgspec resolves struct annotations at runtime.
from src.core.product import OAuthProvider


class OAuthFlow(msgspec.Struct, frozen=True, kw_only=True):
    """Pending authorize→callback flow. Key ``oauth:flow:{state}``."""

    product_id: str
    provider: OAuthProvider
    code_verifier: str
    nonce: str
    binding_hash: str
    return_to: str | None


class OAuthHandoff(msgspec.Struct, frozen=True, kw_only=True):
    """One-time login handoff. Key ``oauth:handoff:{code}``."""

    product_id: str
    user_id: UUID
    binding_hash: str


class PendingSignup(msgspec.Struct, frozen=True, kw_only=True):
    """Verified-but-unregistered identity awaiting legal acceptance. Key ``oauth:signup:{ticket}``.

    Lives only in Redis until complete-signup — no personal data is written
    to the database before the user consents.
    """

    product_id: str
    provider: OAuthProvider
    subject: str
    email: str
    binding_hash: str


class VerifiedIdentity(msgspec.Struct, frozen=True, kw_only=True):
    """Identity extracted from a verified id_token.

    ``email`` is lower-cased and provider-verified by construction — the
    verifier refuses tokens without ``email_verified: true``.
    """

    provider: OAuthProvider
    subject: str
    email: str


@dataclass(frozen=True, slots=True)
class LoginOutcome:
    """The callback resolved to an existing (possibly just-linked) account."""

    user_id: UUID


@dataclass(frozen=True, slots=True)
class SignupOutcome:
    """The callback resolved to an unknown identity — two-step signup follows."""

    identity: VerifiedIdentity


CallbackOutcome = LoginOutcome | SignupOutcome
