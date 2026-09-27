"""OAuth building blocks: PKCE/return_to (I21), Google client (I3), registry (I18),
Settings validator (I19), product-registry consistency (I20), tx cookie, rate limits."""

from __future__ import annotations

import base64
import hashlib
import json
from typing import Any

import httpx
import pytest
import structlog
from pydantic import SecretStr, ValidationError

from src.api.middleware.rate_limit import build_rate_limit_config
from src.api.security.oauth_tx_cookie import (
    OAUTH_TX_COOKIE,
    binding_hash,
    clear_oauth_tx_cookie,
    mint_oauth_tx_cookie,
)
from src.api.services.oauth.errors import OAuthFailedError, OAuthProviderNotEnabledError
from src.api.services.oauth.id_token import JwksIdTokenVerifier
from src.api.services.oauth.pkce import (
    generate_opaque_token,
    generate_verifier,
    s256_challenge,
    validate_return_to,
)
from src.api.services.oauth.providers import google
from src.api.services.oauth.providers.google import GoogleOAuthClient
from src.api.services.oauth.registry import OAuthProviderRegistry
from src.core.config import Settings
from src.core.product import AuthMethod, OAuthProvider
from src.core.product_registry import PRODUCT_REGISTRY, SYNTHARA_CONFIG, VEX_CONFIG
from tests.oauth_support import (
    API_PUBLIC_URL,
    CLIENT_ID,
    CLIENT_SECRET,
    RsaSigner,
    jwks,
    oauth_settings,
    query_params,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# PKCE / return_to
# ---------------------------------------------------------------------------


class TestPkce:
    def test_verifier_length_in_rfc_range(self) -> None:
        verifier = generate_verifier()
        assert len(verifier) == 64
        assert 43 <= len(verifier) <= 128

    def test_values_are_unique(self) -> None:
        assert len({generate_opaque_token() for _ in range(50)}) == 50

    def test_challenge_is_unpadded_base64url_sha256(self) -> None:
        verifier = generate_verifier()
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest())
            .rstrip(b"=")
            .decode()
        )
        assert s256_challenge(verifier) == expected
        assert "=" not in s256_challenge(verifier)


class TestReturnTo:
    """I21 — only same-origin paths survive."""

    @pytest.mark.parametrize("value", [None, "/", "/library", "/library?tab=fav&x=1", "/a/b-c_d"])
    def test_accepts(self, value: str | None) -> None:
        assert validate_return_to(value) == value

    @pytest.mark.parametrize(
        "value",
        [
            "//evil.com",
            "/\\evil.com",
            "https://evil.com",
            "evil.com",
            "",
            "/path with space",
            "/tab\there",
            "/trailing-newline\n",
            "/frag#ment",
            "/nul\x00",
            "/" + "a" * 512,
        ],
    )
    def test_rejects(self, value: str) -> None:
        with pytest.raises(ValueError, match="same-origin"):
            validate_return_to(value)

    def test_max_length_boundary(self) -> None:
        value = "/" + "a" * 511
        assert validate_return_to(value) == value


# ---------------------------------------------------------------------------
# Google client
# ---------------------------------------------------------------------------


class _TokenEndpoint:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self.body = body
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if isinstance(self.body, str):
            return httpx.Response(self.status, text=self.body)
        return httpx.Response(self.status, json=self.body)


def _google(endpoint: _TokenEndpoint, verifier: Any = None) -> GoogleOAuthClient:
    http = httpx.AsyncClient(transport=httpx.MockTransport(endpoint.handler))
    return GoogleOAuthClient(
        client_id=CLIENT_ID,
        client_secret=SecretStr(CLIENT_SECRET),
        http_client=http,
        verifier=verifier,
    )


class TestGoogleClient:
    def test_authorization_url(self) -> None:
        """I3 — S256 challenge of the stored verifier, fixed scope/prompt."""
        verifier = generate_verifier()
        url = _google(_TokenEndpoint(200, {})).authorization_url(
            redirect_uri=f"{API_PUBLIC_URL}/v1/auth/oauth/google/callback",
            state="st",
            nonce="nn",
            code_challenge=s256_challenge(verifier),
        )
        assert url.startswith(google.AUTH_URL + "?")
        assert query_params(url) == {
            "response_type": "code",
            "scope": "openid email",
            "client_id": CLIENT_ID,
            "redirect_uri": f"{API_PUBLIC_URL}/v1/auth/oauth/google/callback",
            "state": "st",
            "nonce": "nn",
            "code_challenge": s256_challenge(verifier),
            "code_challenge_method": "S256",
            "prompt": "select_account",
        }

    async def test_exchange_posts_form_and_verifies(self) -> None:
        signer = RsaSigner()
        endpoint = _TokenEndpoint(200, {"id_token": signer.token(nonce="n1")})

        def jwks_or_token(request: httpx.Request) -> httpx.Response:
            if str(request.url) == google.JWKS_URI:
                return httpx.Response(200, json=jwks(signer))
            return endpoint.handler(request)

        http = httpx.AsyncClient(transport=httpx.MockTransport(jwks_or_token))
        verifier = JwksIdTokenVerifier(
            http,
            provider=OAuthProvider.GOOGLE,
            jwks_uri=google.JWKS_URI,
            issuers=google.ISSUERS,
            cache_ttl_seconds=3600,
        )
        client = GoogleOAuthClient(
            client_id=CLIENT_ID,
            client_secret=SecretStr(CLIENT_SECRET),
            http_client=http,
            verifier=verifier,
        )

        identity = await client.exchange_code(
            code="the-code", code_verifier="the-verifier", redirect_uri="https://r/cb", nonce="n1"
        )

        assert identity.email == "person@example.com"
        (request,) = endpoint.requests
        assert str(request.url) == google.TOKEN_URL
        form = dict(httpx.QueryParams(request.content.decode()))
        assert form == {
            "grant_type": "authorization_code",
            "code": "the-code",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uri": "https://r/cb",
            "code_verifier": "the-verifier",
        }

    @pytest.mark.parametrize(
        ("status", "body"),
        [
            (400, {"error": "invalid_grant", "error_description": "Bad code the-code"}),
            (500, "upstream exploded"),
            (200, {"access_token": "only"}),
            (200, ["not", "an", "object"]),
        ],
    )
    async def test_exchange_failures(self, status: int, body: Any) -> None:
        with structlog.testing.capture_logs() as logs, pytest.raises(OAuthFailedError):
            await _google(_TokenEndpoint(status, body)).exchange_code(
                code="the-code", code_verifier="v", redirect_uri="https://r/cb", nonce="n"
            )
        (event,) = [e for e in logs if e["event"] == "auth.oauth.provider_error"]
        assert set(event) <= {"event", "log_level", "provider", "status_code", "error"}
        rendered = json.dumps(logs)
        assert "the-code" not in rendered
        assert "Bad code" not in rendered

    async def test_transport_error(self) -> None:
        def boom(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("down", request=request)

        client = GoogleOAuthClient(
            client_id=CLIENT_ID,
            client_secret=SecretStr(CLIENT_SECRET),
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
            verifier=None,  # type: ignore[arg-type]
        )
        with pytest.raises(OAuthFailedError):
            await client.exchange_code(code="c", code_verifier="v", redirect_uri="r", nonce="n")

    def test_secret_not_in_repr(self) -> None:
        client = _google(_TokenEndpoint(200, {}))
        assert CLIENT_SECRET not in repr(vars(client))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def _registry(settings: Settings) -> OAuthProviderRegistry:
    return OAuthProviderRegistry.build(settings, httpx.AsyncClient(), PRODUCT_REGISTRY.values())


class TestRegistry:
    """I18 — enabled == allowed AND configured."""

    def test_configured_vex_google_is_enabled(self) -> None:
        registry = _registry(oauth_settings())
        assert registry.is_enabled(VEX_CONFIG, AuthMethod.GOOGLE_OAUTH)
        assert isinstance(registry.get(VEX_CONFIG, OAuthProvider.GOOGLE), GoogleOAuthClient)

    def test_unconfigured_product_is_disabled(self) -> None:
        registry = _registry(oauth_settings())
        assert not registry.is_enabled(SYNTHARA_CONFIG, AuthMethod.GOOGLE_OAUTH)
        with pytest.raises(OAuthProviderNotEnabledError):
            registry.get(SYNTHARA_CONFIG, OAuthProvider.GOOGLE)

    def test_nothing_configured(self) -> None:
        registry = _registry(Settings())
        for product in PRODUCT_REGISTRY.values():
            assert registry.is_enabled(product, AuthMethod.EMAIL_PASSWORD)
            assert not registry.is_enabled(product, AuthMethod.GOOGLE_OAUTH)
            assert not registry.is_enabled(product, AuthMethod.APPLE_OAUTH)

    def test_configured_but_not_allowed_is_disabled(self) -> None:
        from dataclasses import replace

        no_google = replace(VEX_CONFIG, allowed_auth_methods=frozenset({AuthMethod.EMAIL_PASSWORD}))
        registry = OAuthProviderRegistry.build(oauth_settings(), httpx.AsyncClient(), [no_google])
        assert not registry.is_enabled(no_google, AuthMethod.GOOGLE_OAUTH)
        with pytest.raises(OAuthProviderNotEnabledError):
            registry.get(no_google, OAuthProvider.GOOGLE)

    def test_apple_never_enabled_yet(self) -> None:
        settings = oauth_settings(
            google_oauth_client_id_synthara="s-id",
            google_oauth_client_secret_synthara="s-secret",
            api_public_url_synthara="https://api.synthara.test",
        )
        registry = _registry(settings)
        assert registry.is_enabled(SYNTHARA_CONFIG, AuthMethod.GOOGLE_OAUTH)
        assert not registry.is_enabled(SYNTHARA_CONFIG, AuthMethod.APPLE_OAUTH)

    def test_redirect_uri_from_api_public_url(self) -> None:
        registry = _registry(oauth_settings())
        assert (
            registry.redirect_uri(VEX_CONFIG, OAuthProvider.GOOGLE)
            == f"{API_PUBLIC_URL}/v1/auth/oauth/google/callback"
        )

    def test_build_logs_no_secrets(self) -> None:
        with structlog.testing.capture_logs() as logs:
            _registry(oauth_settings())
        events = {e["event"] for e in logs}
        assert {"auth.oauth.provider_enabled", "auth.oauth.provider_disabled"} <= events
        assert CLIENT_SECRET not in json.dumps(logs, default=str)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class TestSettingsValidator:
    """I19 — half-configured clients fail startup."""

    @pytest.mark.parametrize(
        ("overrides", "match"),
        [
            ({"google_oauth_client_secret_vex": None}, "must be set together"),
            ({"google_oauth_client_id_vex": None}, "must be set together"),
            ({"api_public_url_vex": None}, "API_PUBLIC_URL_VEX"),
            ({"redis_url": None}, "REDIS_URL"),
            ({"api_public_url_vex": "https://api.vex.test/"}, "bare origin"),
            ({"api_public_url_vex": "api.vex.test"}, "bare origin"),
        ],
    )
    def test_rejects(self, overrides: dict[str, Any], match: str) -> None:
        with pytest.raises(ValidationError, match=match):
            oauth_settings(**overrides)

    def test_synthara_client_needs_its_own_public_url(self) -> None:
        with pytest.raises(ValidationError, match="API_PUBLIC_URL_SYNTHARA"):
            oauth_settings(
                google_oauth_client_id_synthara="s-id",
                google_oauth_client_secret_synthara="s-secret",
            )

    def test_unconfigured_is_valid(self) -> None:
        settings = Settings(redis_url=None)
        assert settings.google_oauth_client_id_vex is None

    def test_fully_configured_is_valid(self) -> None:
        settings = oauth_settings()
        assert settings.api_public_url_for("vex") == API_PUBLIC_URL

    def test_secret_is_masked(self) -> None:
        assert CLIENT_SECRET not in repr(oauth_settings())

    def test_url_helpers_are_strict(self) -> None:
        settings = oauth_settings()
        assert settings.app_url_for("vex") == settings.app_url_vex
        assert settings.app_url_for("synthara") == settings.app_url_synthara
        assert settings.api_public_url_for("synthara") is None
        with pytest.raises(KeyError):
            settings.app_url_for("apex")
        with pytest.raises(KeyError):
            settings.api_public_url_for("apex")


# ---------------------------------------------------------------------------
# Product registry consistency (I20)
# ---------------------------------------------------------------------------

# Apple sign-in is allowed for Synthara but not wired yet — the named
# follow-up "OAuth: Apple" arc (form_post callback, SameSite=None binding
# cookie) removes this entry. Any other gap fails the build.
_OAUTH_NOT_YET_IMPLEMENTED = frozenset({AuthMethod.APPLE_OAUTH})
_OAUTH_METHODS = frozenset(AuthMethod) - {AuthMethod.EMAIL_PASSWORD}


class TestProductOAuthConsistency:
    def test_oauth_provider_auth_method_total(self) -> None:
        for provider in OAuthProvider:
            assert isinstance(provider.auth_method, AuthMethod)
            assert provider.auth_method in _OAUTH_METHODS

    @pytest.mark.parametrize("product", list(PRODUCT_REGISTRY.values()), ids=lambda p: p.slug)
    def test_allowed_oauth_methods_match_clients(self, product: Any) -> None:
        providers = [c.provider for c in product.oauth_clients]
        assert len(providers) == len(set(providers)), "at most one client per provider"
        allowed = (product.allowed_auth_methods & _OAUTH_METHODS) - _OAUTH_NOT_YET_IMPLEMENTED
        assert allowed == {p.auth_method for p in providers}

    @pytest.mark.parametrize("product", list(PRODUCT_REGISTRY.values()), ids=lambda p: p.slug)
    def test_client_env_names_exist_on_settings(self, product: Any) -> None:
        fields = Settings.model_fields
        for client in product.oauth_clients:
            assert client.client_id_env in fields
            assert client.client_secret_env in fields


# ---------------------------------------------------------------------------
# Tx cookie / rate limits
# ---------------------------------------------------------------------------


class TestTxCookie:
    def test_mint_flags(self) -> None:
        cookie = mint_oauth_tx_cookie("binding", max_age=900, secure=True)
        assert cookie.key == OAUTH_TX_COOKIE == "apex_oauth_tx"
        assert cookie.httponly is True
        assert cookie.secure is True
        assert cookie.samesite == "lax"
        assert cookie.path == "/v1/auth/oauth"
        assert cookie.domain is None
        assert cookie.max_age == 900

    def test_clear(self) -> None:
        cookie = clear_oauth_tx_cookie(secure=False)
        assert cookie.max_age == 0
        assert cookie.value == ""
        assert cookie.path == "/v1/auth/oauth"
        assert cookie.domain is None

    def test_binding_hash(self) -> None:
        assert binding_hash("abc") == hashlib.sha256(b"abc").hexdigest()


class TestRateLimits:
    def test_oauth_routes_limited(self) -> None:
        settings = Settings()
        config = build_rate_limit_config(settings)
        for provider in OAuthProvider:
            for step in ("authorize", "callback"):
                key = f"GET /v1/auth/oauth/{provider.value}/{step}"
                assert config[key] == settings.rate_limit_oauth_authorize
        assert config["POST /v1/auth/oauth/exchange"] == settings.rate_limit_oauth_exchange
        assert config["POST /v1/auth/oauth/signup-info"] == settings.rate_limit_oauth_exchange
        assert (
            config["POST /v1/auth/oauth/complete-signup"]
            == settings.rate_limit_oauth_complete_signup
        )
        assert settings.rate_limit_oauth_complete_signup == settings.rate_limit_register
