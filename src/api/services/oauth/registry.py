"""Which OAuth providers are usable for which product, and their clients.

Built once in ``init_services``. A provider is **enabled** for a product only
when its ``AuthMethod`` is in ``product.allowed_auth_methods`` AND the product
has an ``OAuthClientEnv`` for it whose id and secret are both set.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, assert_never

import structlog

from src.api.services.oauth.errors import OAuthProviderNotEnabledError
from src.api.services.oauth.id_token import JwksIdTokenVerifier
from src.api.services.oauth.providers import google
from src.core.product import AuthMethod, OAuthProvider

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    import httpx
    from pydantic import SecretStr

    from src.api.services.oauth.providers.base import OAuthProviderClient
    from src.core.config import Settings
    from src.core.product import ProductConfig

logger = structlog.get_logger(__name__)


class OAuthProviderRegistry:
    """Resolves ``(product, provider)`` to a configured client.

    Args:
        settings: Supplies ``api_public_url_{slug}`` for redirect URIs.
        clients: Enabled clients keyed by ``(product slug, provider)``.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        clients: Mapping[tuple[str, OAuthProvider], OAuthProviderClient],
    ) -> None:
        self._settings = settings
        self._clients = dict(clients)

    @classmethod
    def build(
        cls,
        settings: Settings,
        http_client: httpx.AsyncClient,
        products: Iterable[ProductConfig],
    ) -> OAuthProviderRegistry:
        """Construct clients for every allowed-and-configured (product, provider).

        One JWKS verifier per provider is shared across products (the key set
        is the provider's, not the client's). Logs enabled/disabled once per
        pair — never any credential.
        """
        verifiers: dict[OAuthProvider, JwksIdTokenVerifier] = {}
        clients: dict[tuple[str, OAuthProvider], OAuthProviderClient] = {}
        for product in products:
            for client_env in product.oauth_clients:
                provider = client_env.provider
                client_id: str | None = getattr(settings, client_env.client_id_env)
                client_secret: SecretStr | None = getattr(settings, client_env.client_secret_env)
                allowed = provider.auth_method in product.allowed_auth_methods
                if not (allowed and client_id and client_secret):
                    logger.info(
                        "auth.oauth.provider_disabled",
                        product_id=product.slug,
                        provider=provider.value,
                        allowed=allowed,
                        configured=bool(client_id and client_secret),
                    )
                    continue
                if provider not in verifiers:
                    verifiers[provider] = _build_verifier(provider, settings, http_client)
                clients[(product.slug, provider)] = _build_client(
                    provider,
                    client_id=client_id,
                    client_secret=client_secret,
                    http_client=http_client,
                    verifier=verifiers[provider],
                )
                logger.info(
                    "auth.oauth.provider_enabled", product_id=product.slug, provider=provider.value
                )
        return cls(settings=settings, clients=clients)

    def get(self, product: ProductConfig, provider: OAuthProvider) -> OAuthProviderClient:
        """The enabled client for a product/provider.

        Raises:
            OAuthProviderNotEnabledError: Not allowed or not configured (404).
        """
        if provider.auth_method not in product.allowed_auth_methods:
            raise OAuthProviderNotEnabledError(provider.value)
        client = self._clients.get((product.slug, provider))
        if client is None:
            raise OAuthProviderNotEnabledError(provider.value)
        return client

    def is_enabled(self, product: ProductConfig, method: AuthMethod) -> bool:
        """Whether an auth method is both allowed and usable for the product.

        ``EMAIL_PASSWORD`` needs no configuration; an OAuth method needs an
        enabled client. A method with no ``OAuthProvider`` yet (Apple) is
        never enabled.
        """
        if method not in product.allowed_auth_methods:
            return False
        if method is AuthMethod.EMAIL_PASSWORD:
            return True
        return any(
            provider.auth_method is method and (product.slug, provider) in self._clients
            for provider in OAuthProvider
        )

    def redirect_uri(self, product: ProductConfig, provider: OAuthProvider) -> str:
        """The single place the redirect_uri is built — must byte-match the GCP registration.

        Built from ``api_public_url_{slug}``, never from request headers.

        Raises:
            RuntimeError: No public API URL (unreachable for an enabled client —
                ``Settings.validate_oauth_config`` requires it).
        """
        base = self._settings.api_public_url_for(product.slug)
        if not base:
            raise RuntimeError(f"api_public_url_{product.slug} is not configured")
        return f"{base}/v1/auth/oauth/{provider.value}/callback"


def _build_verifier(
    provider: OAuthProvider, settings: Settings, http_client: httpx.AsyncClient
) -> JwksIdTokenVerifier:
    match provider:
        case OAuthProvider.GOOGLE:
            return JwksIdTokenVerifier(
                http_client,
                provider=provider,
                jwks_uri=google.JWKS_URI,
                issuers=google.ISSUERS,
                cache_ttl_seconds=settings.oauth_jwks_cache_ttl_seconds,
            )
        case _:
            assert_never(provider)


def _build_client(
    provider: OAuthProvider,
    *,
    client_id: str,
    client_secret: SecretStr,
    http_client: httpx.AsyncClient,
    verifier: JwksIdTokenVerifier,
) -> OAuthProviderClient:
    match provider:
        case OAuthProvider.GOOGLE:
            return google.GoogleOAuthClient(
                client_id=client_id,
                client_secret=client_secret,
                http_client=http_client,
                verifier=verifier,
            )
        case _:
            assert_never(provider)
