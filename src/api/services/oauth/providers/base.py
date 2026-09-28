"""The provider-client seam: one implementation per OAuth/OIDC provider."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.api.services.oauth.models import VerifiedIdentity
    from src.core.product import OAuthProvider


class OAuthProviderClient(Protocol):
    """Builds the authorization URL and redeems a code for a verified identity."""

    @property
    def provider(self) -> OAuthProvider:
        """Which provider this client talks to."""
        ...

    def authorization_url(
        self, *, redirect_uri: str, state: str, nonce: str, code_challenge: str
    ) -> str:
        """The provider consent URL for one flow."""
        ...

    async def exchange_code(
        self, *, code: str, code_verifier: str, redirect_uri: str, nonce: str
    ) -> VerifiedIdentity:
        """Redeem an authorization code and verify the returned id_token.

        Raises:
            OAuthFailedError: Token endpoint or id_token verification failed.
            EmailUnverifiedError: The provider did not verify the email.
        """
        ...
