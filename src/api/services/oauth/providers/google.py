"""Google OIDC client (authorization-code flow, confidential client + PKCE).

Endpoints are hardcoded ``Final`` constants rather than read from the
discovery document: Google's are stable, and runtime discovery would add a
startup network dependency for no benefit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final
from urllib.parse import urlencode

import httpx
import structlog

from src.api.services.oauth.errors import OAuthFailedError
from src.core.product import OAuthProvider

if TYPE_CHECKING:
    from pydantic import SecretStr

    from src.api.services.oauth.id_token import JwksIdTokenVerifier
    from src.api.services.oauth.models import VerifiedIdentity

logger = structlog.get_logger(__name__)

AUTH_URL: Final = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL: Final = "https://oauth2.googleapis.com/token"  # noqa: S105 — endpoint, not a secret
JWKS_URI: Final = "https://www.googleapis.com/oauth2/v3/certs"
ISSUERS: Final = frozenset({"https://accounts.google.com", "accounts.google.com"})
SCOPE: Final = "openid email"


class GoogleOAuthClient:
    """One product's Google OAuth client (each brand has its own GCP project).

    Args:
        client_id: OAuth client id (also the expected ``aud``/``azp``).
        client_secret: Unwrapped only while building the token request body.
        http_client: Shared async client.
        verifier: Google's JWKS id_token verifier (shared across products).
    """

    def __init__(
        self,
        *,
        client_id: str,
        client_secret: SecretStr,
        http_client: httpx.AsyncClient,
        verifier: JwksIdTokenVerifier,
    ) -> None:
        self._client_id = client_id
        self._client_secret = client_secret
        self._http = http_client
        self._verifier = verifier

    @property
    def provider(self) -> OAuthProvider:
        """Always Google."""
        return OAuthProvider.GOOGLE

    def authorization_url(
        self, *, redirect_uri: str, state: str, nonce: str, code_challenge: str
    ) -> str:
        """Google consent URL for one flow."""
        params = {
            "response_type": "code",
            "scope": SCOPE,
            "client_id": self._client_id,
            "redirect_uri": redirect_uri,
            "state": state,
            "nonce": nonce,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }
        return f"{AUTH_URL}?{urlencode(params)}"

    async def exchange_code(
        self, *, code: str, code_verifier: str, redirect_uri: str, nonce: str
    ) -> VerifiedIdentity:
        """Redeem the code at the token endpoint and verify the id_token.

        Raises:
            OAuthFailedError: Transport error, non-2xx, missing id_token, or
                id_token verification failure.
            EmailUnverifiedError: Google did not verify the email.
        """
        try:
            response = await self._http.post(
                TOKEN_URL,
                data={
                    "grant_type": "authorization_code",
                    "code": code,
                    "client_id": self._client_id,
                    "client_secret": self._client_secret.get_secret_value(),
                    "redirect_uri": redirect_uri,
                    "code_verifier": code_verifier,
                },
                headers={"Accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            logger.warning(
                "auth.oauth.provider_error",
                provider=self.provider.value,
                error_type=type(exc).__name__,
            )
            raise OAuthFailedError from exc

        body = _json_object(response)
        if not response.is_success:
            # Only the status and the provider's short error code — never the
            # body wholesale (it can echo request parameters).
            error = body.get("error") if body is not None else None
            logger.warning(
                "auth.oauth.provider_error",
                provider=self.provider.value,
                status_code=response.status_code,
                error=error if isinstance(error, str) else None,
            )
            raise OAuthFailedError

        id_token = body.get("id_token") if body is not None else None
        if not isinstance(id_token, str) or not id_token:
            logger.warning(
                "auth.oauth.provider_error",
                provider=self.provider.value,
                status_code=response.status_code,
                error="missing_id_token",
            )
            raise OAuthFailedError

        return await self._verifier.verify(id_token, audience=self._client_id, nonce=nonce)


def _json_object(response: httpx.Response) -> dict[str, object] | None:
    try:
        body = response.json()
    except ValueError:
        return None
    return body if isinstance(body, dict) else None
