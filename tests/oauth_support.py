"""Shared builders for OAuth tests (unit + integration).

- ``RsaSigner``: a local RSA key that signs id_tokens and publishes its JWKS.
- ``InMemoryOAuthFlowStore``: the ``OAuthFlowStore`` Protocol without Redis
  (``take_*`` is atomic — no ``await`` between read and delete).
- ``FakeProviderClient``: an ``OAuthProviderClient`` that returns a preset
  identity (or raises), recording what it was called with.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlencode, urlsplit

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from jwt.algorithms import RSAAlgorithm

from src.api.services.oauth.models import (
    OAuthFlow,
    OAuthHandoff,
    PendingSignup,
    VerifiedIdentity,
)
from src.api.services.oauth.registry import OAuthProviderRegistry
from src.core.config import Settings
from src.core.product import OAuthProvider
from src.core.uid import new_id
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from uuid import UUID

    from pydantic import SecretStr

GOOGLE_ISSUER = "https://accounts.google.com"
CLIENT_ID = "vex-client.apps.googleusercontent.com"
CLIENT_SECRET = "vex-client-secret-value"
API_PUBLIC_URL = "https://api.vex.test"
APP_URL = "https://vex.test"


# ---------------------------------------------------------------------------
# id_token signing
# ---------------------------------------------------------------------------


class RsaSigner:
    """One RSA signing key with a ``kid``; mints id_tokens and exposes its JWKS."""

    def __init__(self, kid: str = "key-1") -> None:
        self.kid = kid
        self._key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    def jwk(self) -> dict[str, Any]:
        public = json.loads(RSAAlgorithm.to_jwk(self._key.public_key()))
        public.update({"kid": self.kid, "alg": "RS256", "use": "sig"})
        return dict(public)

    def token(self, **overrides: Any) -> str:
        """An id_token with valid defaults; ``overrides`` replace claims (``None`` removes)."""
        now = int(time.time())
        claims: dict[str, Any] = {
            "iss": GOOGLE_ISSUER,
            "aud": CLIENT_ID,
            "azp": CLIENT_ID,
            "sub": "google-subject-123",
            "email": "Person@Example.com",
            "email_verified": True,
            "nonce": "expected-nonce",
            "iat": now,
            "exp": now + 3600,
        }
        headers = {"kid": overrides.pop("_kid", self.kid)}
        for name, value in overrides.items():
            if value is None:
                claims.pop(name, None)
            else:
                claims[name] = value
        return jwt.encode(claims, self._key, algorithm="RS256", headers=headers)


def jwks(*signers: RsaSigner) -> dict[str, Any]:
    return {"keys": [s.jwk() for s in signers]}


# ---------------------------------------------------------------------------
# Flow store
# ---------------------------------------------------------------------------


class InMemoryOAuthFlowStore:
    """``OAuthFlowStore`` over dicts. TTLs are recorded but not enforced."""

    def __init__(self) -> None:
        self.flows: dict[str, OAuthFlow] = {}
        self.handoffs: dict[str, OAuthHandoff] = {}
        self.signups: dict[str, PendingSignup] = {}
        self.ttls: dict[str, int] = {}

    async def put_flow(self, state: str, flow: OAuthFlow, ttl: int) -> None:
        self.flows[state] = flow
        self.ttls[f"flow:{state}"] = ttl

    async def take_flow(self, state: str) -> OAuthFlow | None:
        return self.flows.pop(state, None)

    async def put_handoff(self, code: str, handoff: OAuthHandoff, ttl: int) -> None:
        self.handoffs[code] = handoff
        self.ttls[f"handoff:{code}"] = ttl

    async def take_handoff(self, code: str) -> OAuthHandoff | None:
        return self.handoffs.pop(code, None)

    async def put_signup(self, ticket: str, signup: PendingSignup, ttl: int) -> None:
        self.signups[ticket] = signup
        self.ttls[f"signup:{ticket}"] = ttl

    async def peek_signup(self, ticket: str) -> PendingSignup | None:
        return self.signups.get(ticket)

    async def take_signup(self, ticket: str) -> PendingSignup | None:
        return self.signups.pop(ticket, None)


async def commit_active_user_for_email_race(
    engine: Any, *, email: str, product_id: str = "vex"
) -> UUID:
    """Commit the competing row from another session for signup race tests."""
    from sqlalchemy.ext.asyncio import AsyncSession

    user_id = new_id()
    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await UserRepository(session).create_user(
            id=user_id,
            email=email,
            password_hash="race-password-hash",
            product_id=product_id,
        )
        await session.commit()
    return user_id


async def delete_email_race_user(engine: Any, user_id: UUID) -> None:
    """Remove the separately committed race fixture after its test."""
    from sqlalchemy import delete
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User

    async with AsyncSession(bind=engine, expire_on_commit=False) as session:
        await session.execute(delete(User).where(User.id == user_id))
        await session.commit()


# ---------------------------------------------------------------------------
# Provider client
# ---------------------------------------------------------------------------


@dataclass
class FakeProviderClient:
    """Returns ``identity`` from ``exchange_code`` (or raises ``error``)."""

    identity: VerifiedIdentity | None = None
    error: Exception | None = None
    provider: OAuthProvider = OAuthProvider.GOOGLE
    exchanges: list[dict[str, str]] = field(default_factory=list)

    def authorization_url(
        self, *, redirect_uri: str, state: str, nonce: str, code_challenge: str
    ) -> str:
        query = urlencode(
            {
                "redirect_uri": redirect_uri,
                "state": state,
                "nonce": nonce,
                "code_challenge": code_challenge,
            }
        )
        return f"https://provider.test/auth?{query}"

    async def exchange_code(
        self, *, code: str, code_verifier: str, redirect_uri: str, nonce: str
    ) -> VerifiedIdentity:
        self.exchanges.append(
            {
                "code": code,
                "code_verifier": code_verifier,
                "redirect_uri": redirect_uri,
                "nonce": nonce,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.identity is not None
        return self.identity


def identity(
    subject: str = "google-subject-123", email: str = "person@example.com"
) -> VerifiedIdentity:
    return VerifiedIdentity(provider=OAuthProvider.GOOGLE, subject=subject, email=email)


# ---------------------------------------------------------------------------
# Settings / registry
# ---------------------------------------------------------------------------


def oauth_settings(**overrides: Any) -> Settings:
    """Settings with a configured vex Google client (debug → non-Secure cookies)."""
    values: dict[str, Any] = {
        "debug": True,
        "redis_url": "redis://localhost:6379/15",
        "google_oauth_client_id_vex": CLIENT_ID,
        "google_oauth_client_secret_vex": CLIENT_SECRET,
        "api_public_url_vex": API_PUBLIC_URL,
        "app_url_vex": APP_URL,
    }
    values.update(overrides)
    return Settings(**values)


def fake_registry(
    settings: Settings, client: FakeProviderClient, *, product_slug: str = "vex"
) -> OAuthProviderRegistry:
    return OAuthProviderRegistry(
        settings=settings, clients={(product_slug, client.provider): client}
    )


def secret_value(secret: SecretStr | None) -> str:
    assert secret is not None
    return secret.get_secret_value()


# ---------------------------------------------------------------------------
# URL parsing
# ---------------------------------------------------------------------------


def query_params(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}


def fragment_params(url: str) -> dict[str, str]:
    return {k: v[0] for k, v in parse_qs(urlsplit(url).fragment).items()}
