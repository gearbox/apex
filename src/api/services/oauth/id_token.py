"""OIDC ``id_token`` verification against a provider's JWKS.

JWKS is fetched with the injected async ``httpx`` client — never PyJWT's
built-in JWKS client, which does blocking I/O on the event loop. Keys are
cached by ``kid`` for ``cache_ttl_seconds``; an unknown ``kid`` (key
rotation) triggers one forced refetch, serialized by a lock and throttled to
at most one per ``_FORCED_REFETCH_MIN_INTERVAL_SECONDS`` so a flood of
tokens with bogus ``kid`` values can't turn into a flood of JWKS requests.

Never logs ``sub``, ``email``, the token, or the nonce.
"""

from __future__ import annotations

import asyncio
import hmac
import time
from typing import TYPE_CHECKING, Any, Final

import httpx
import jwt
import structlog

from src.api.services.oauth.errors import EmailUnverifiedError, OAuthFailedError
from src.api.services.oauth.models import VerifiedIdentity

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.product import OAuthProvider

logger = structlog.get_logger(__name__)

_FORCED_REFETCH_MIN_INTERVAL_SECONDS: Final = 60.0
_LEEWAY_SECONDS: Final = 30
_REQUIRED_CLAIMS: Final = ["exp", "iat", "iss", "aud", "sub", "nonce"]
# users.email and user_identities.subject are VARCHAR(255).
_MAX_CLAIM_LENGTH: Final = 255


class JwksIdTokenVerifier:
    """Verifies RS256 id_tokens for one provider (shared across products).

    Args:
        http_client: Shared async client (owns timeouts).
        provider: Provider stamped onto the returned identity.
        jwks_uri: The provider's JWKS endpoint.
        issuers: Accepted ``iss`` values.
        cache_ttl_seconds: How long a fetched key set is trusted.
        clock: Monotonic clock (injectable for tests).
    """

    def __init__(
        self,
        http_client: httpx.AsyncClient,
        *,
        provider: OAuthProvider,
        jwks_uri: str,
        issuers: frozenset[str],
        cache_ttl_seconds: int,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._http = http_client
        self._provider = provider
        self._jwks_uri = jwks_uri
        self._issuers = issuers
        self._cache_ttl = float(cache_ttl_seconds)
        self._clock = clock
        self._keys: dict[str, jwt.PyJWK] = {}
        self._fetched_at: float | None = None
        self._last_forced_at: float | None = None
        self._lock = asyncio.Lock()

    async def verify(self, id_token: str, *, audience: str, nonce: str) -> VerifiedIdentity:
        """Verify signature and claims, returning the identity.

        Raises:
            OAuthFailedError: Bad signature/header/kid, wrong ``aud``/``iss``/
                ``azp``, expired, nonce mismatch, missing ``email``, or a
                non-boolean ``email_verified``.
            EmailUnverifiedError: ``email_verified`` missing or ``False``.
        """
        try:
            header = jwt.get_unverified_header(id_token)
        except jwt.PyJWTError as exc:
            self._reject("malformed_header")
            raise OAuthFailedError from exc
        kid = header.get("kid")
        if not isinstance(kid, str) or not kid:
            self._reject("missing_kid")
            raise OAuthFailedError

        key = await self._key_for(kid)
        try:
            claims: dict[str, Any] = jwt.decode(
                id_token,
                key=key,
                algorithms=["RS256"],
                audience=audience,
                issuer=list(self._issuers),
                options={"require": _REQUIRED_CLAIMS},
                leeway=_LEEWAY_SECONDS,
            )
        except jwt.PyJWTError as exc:
            self._reject(type(exc).__name__)
            raise OAuthFailedError from exc

        return self._identity_from_claims(claims, audience=audience, nonce=nonce)

    def _identity_from_claims(
        self, claims: dict[str, Any], *, audience: str, nonce: str
    ) -> VerifiedIdentity:
        token_nonce = claims.get("nonce")
        if not isinstance(token_nonce, str) or not hmac.compare_digest(
            token_nonce.encode(), nonce.encode()
        ):
            self._reject("nonce_mismatch")
            raise OAuthFailedError
        if "azp" in claims and claims["azp"] != audience:
            self._reject("azp_mismatch")
            raise OAuthFailedError

        subject = claims.get("sub")
        if not isinstance(subject, str) or not subject or len(subject) > _MAX_CLAIM_LENGTH:
            self._reject("invalid_sub")
            raise OAuthFailedError
        email = claims.get("email")
        if not isinstance(email, str) or not email or len(email) > _MAX_CLAIM_LENGTH:
            self._reject("missing_email")
            raise OAuthFailedError

        # Strictly the JSON boolean true: a string "true" is a malformed
        # token (fail), a missing/false value is an honest "unverified".
        email_verified = claims.get("email_verified")
        if email_verified is None or email_verified is False:
            self._reject("email_unverified")
            raise EmailUnverifiedError
        if email_verified is not True:
            self._reject("email_verified_not_boolean")
            raise OAuthFailedError

        return VerifiedIdentity(provider=self._provider, subject=subject, email=email.lower())

    async def _key_for(self, kid: str) -> jwt.PyJWK:
        fetched_now = False
        if self._is_stale():
            fetched_now = await self._refresh(force=False)
        key = self._keys.get(kid)
        if key is None and not fetched_now:
            await self._refresh(force=True)
            key = self._keys.get(kid)
        if key is None:
            self._reject("unknown_kid")
            raise OAuthFailedError
        return key

    def _is_stale(self) -> bool:
        return self._fetched_at is None or self._clock() - self._fetched_at >= self._cache_ttl

    async def _refresh(self, *, force: bool) -> bool:
        """Fetch the key set; returns whether a fetch happened (vs. skipped/throttled)."""
        async with self._lock:
            now = self._clock()
            if force:
                if (
                    self._last_forced_at is not None
                    and now - self._last_forced_at < _FORCED_REFETCH_MIN_INTERVAL_SECONDS
                ):
                    return False
                self._last_forced_at = now
            elif not self._is_stale():
                # Another coroutine refreshed while we waited for the lock.
                return False
            self._keys = await self._fetch()
            self._fetched_at = self._clock()
            return True

    async def _fetch(self) -> dict[str, jwt.PyJWK]:
        try:
            response = await self._http.get(self._jwks_uri)
        except httpx.HTTPError as exc:
            logger.warning(
                "auth.oauth.jwks_fetch_failed",
                provider=self._provider.value,
                error_type=type(exc).__name__,
            )
            raise OAuthFailedError from exc
        if not response.is_success:
            logger.warning(
                "auth.oauth.jwks_fetch_failed",
                provider=self._provider.value,
                status_code=response.status_code,
            )
            raise OAuthFailedError
        try:
            key_set = jwt.PyJWKSet.from_dict(response.json())
        except (ValueError, jwt.PyJWKSetError, jwt.PyJWKError) as exc:
            logger.warning(
                "auth.oauth.jwks_invalid",
                provider=self._provider.value,
                error_type=type(exc).__name__,
            )
            raise OAuthFailedError from exc
        return {key.key_id: key for key in key_set.keys if key.key_id}

    def _reject(self, reason: str) -> None:
        logger.warning("auth.oauth.id_token_rejected", provider=self._provider.value, reason=reason)
