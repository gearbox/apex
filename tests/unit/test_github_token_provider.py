"""Unit tests for the GitHub credential providers (App installation token and static PAT)."""

from __future__ import annotations

import asyncio
import inspect
import json
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

import httpx
import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from src.api.dependencies.common import build_github_token_provider
from src.api.services.github.token_provider import (
    GitHubAppTokenProvider,
    GitHubToken,
    GitHubTokenUnavailableError,
    StaticGitHubTokenProvider,
)
from src.core.enums import GitHubAuthMode
from tests.unit.helpers import hermetic_settings

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

_NOW = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
_MARGIN = timedelta(seconds=600)
_MINT_URL = "https://api.github.com/app/installations/678/access_tokens"
_JWT_SECRET = "x" * 48


@pytest.fixture(scope="module")
def private_key() -> rsa.RSAPrivateKey:
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


@pytest.fixture(scope="module")
def private_pem(private_key: rsa.RSAPrivateKey) -> bytes:
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@pytest.fixture(scope="module")
def public_pem(private_key: rsa.RSAPrivateKey) -> bytes:
    return private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
    )


class _Clock:
    def __init__(self, now: datetime = _NOW) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now


def _iso(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%SZ")


def _created(token: str = "ghs_minted_1", expires_at: datetime | None = None) -> httpx.Response:
    return httpx.Response(
        201,
        json={
            "token": token,
            "expires_at": _iso(expires_at or _NOW + timedelta(hours=1)),
            "permissions": {"contents": "read"},
        },
    )


class _Upstream:
    """A recording MockTransport; ``respond`` decides each response."""

    def __init__(
        self, respond: Callable[[httpx.Request], httpx.Response | Awaitable[httpx.Response]]
    ) -> None:
        self.requests: list[httpx.Request] = []
        self._respond = respond

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        result = self._respond(request)
        return await result if inspect.isawaitable(result) else result

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self))


def _provider(
    private_pem: bytes, upstream: _Upstream, clock: _Clock | None = None
) -> GitHubAppTokenProvider:
    return GitHubAppTokenProvider(
        app_id=12345,
        installation_id=678,
        private_key_pem=private_pem,
        repositories=("aisha", "ai-bundles"),
        http_client=upstream.client(),
        refresh_margin=_MARGIN,
        clock=clock or _Clock(),
    )


def _bearer_jwt(request: httpx.Request) -> str:
    scheme, _, value = request.headers["authorization"].partition(" ")
    assert scheme == "Bearer"
    return value


class TestMintRequest:
    """A4 — the exact JWT, URL, headers and body GitHub expects."""

    async def test_request_shape_and_returned_token(
        self, private_pem: bytes, public_pem: bytes
    ) -> None:
        upstream = _Upstream(lambda _r: _created("ghs_minted_1"))
        provider = _provider(private_pem, upstream)

        token = await provider.get_token()

        assert provider.mode is GitHubAuthMode.app
        assert token.value == "ghs_minted_1"
        assert token.expires_at == _NOW + timedelta(hours=1)
        assert token.expires_at is not None
        assert token.expires_at.tzinfo is not None

        assert len(upstream.requests) == 1
        request = upstream.requests[0]
        assert request.method == "POST"
        assert str(request.url) == _MINT_URL
        assert request.headers["accept"] == "application/vnd.github+json"
        assert request.headers["x-github-api-version"] == "2022-11-28"
        assert json.loads(request.content) == {
            "repositories": ["aisha", "ai-bundles"],
            "permissions": {"contents": "read"},
        }

        encoded = _bearer_jwt(request)
        assert jwt.get_unverified_header(encoded)["alg"] == "RS256"
        # Signature verified with the public half; time claims are asserted below
        # against the injected clock rather than the wall clock.
        claims = jwt.decode(
            encoded,
            public_pem,
            algorithms=["RS256"],
            options={"verify_exp": False, "verify_iat": False},
        )
        now = int(_NOW.timestamp())
        assert claims == {"iss": "12345", "iat": now - 60, "exp": now + 540}
        assert claims["exp"] - claims["iat"] <= 600  # GitHub rejects a longer window

    async def test_expires_at_comes_from_github_not_the_local_clock(
        self, private_pem: bytes
    ) -> None:
        upstream = _Upstream(lambda _r: _created(expires_at=_NOW + timedelta(minutes=17)))
        provider = _provider(private_pem, upstream)

        token = await provider.get_token()

        assert token.expires_at == _NOW + timedelta(minutes=17)


class TestCaching:
    """A5 — reuse while more than the margin remains; refresh once it drops to/below it."""

    async def test_reused_while_margin_remains_then_refreshed(self, private_pem: bytes) -> None:
        issued = iter(["ghs_minted_1", "ghs_minted_2"])
        clock = _Clock()
        upstream = _Upstream(
            lambda _r: _created(next(issued), expires_at=clock.now + timedelta(hours=1))
        )
        provider = _provider(private_pem, upstream, clock)

        first = await provider.get_token()
        # 601 s left > 600 s margin: still cached.
        clock.now = _NOW + timedelta(hours=1) - timedelta(seconds=601)
        assert await provider.get_token() is first
        assert len(upstream.requests) == 1

        # Exactly the margin left is *not* "more than": refresh.
        clock.now = _NOW + timedelta(hours=1) - _MARGIN
        second = await provider.get_token()

        assert second.value == "ghs_minted_2"
        assert len(upstream.requests) == 2
        # And the refreshed token is cached in turn.
        assert await provider.get_token() is second
        assert len(upstream.requests) == 2

    async def test_expired_token_is_never_served(self, private_pem: bytes) -> None:
        clock = _Clock()
        responses = iter([_created("ghs_minted_1"), httpx.Response(502, text="bad gateway")])
        upstream = _Upstream(lambda _r: next(responses))
        provider = _provider(private_pem, upstream, clock)

        await provider.get_token()
        clock.now = _NOW + timedelta(hours=2)

        with pytest.raises(GitHubTokenUnavailableError):
            await provider.get_token()


class TestSingleFlight:
    """A6 — a cold-cache stampede mints exactly once."""

    async def test_ten_concurrent_callers_cause_one_mint(self, private_pem: bytes) -> None:
        async def slow_created(_request: httpx.Request) -> httpx.Response:
            await asyncio.sleep(0.05)  # keep the first mint in flight while the rest queue
            return _created("ghs_minted_1")

        upstream = _Upstream(slow_created)
        provider = _provider(private_pem, upstream)

        tokens = await asyncio.gather(*(provider.get_token() for _ in range(10)))

        assert len(upstream.requests) == 1
        assert {token.value for token in tokens} == {"ghs_minted_1"}


class TestFailures:
    """A7 — every failure is a GitHubTokenUnavailableError with a secret-free message."""

    @staticmethod
    def _assert_message_is_clean(
        error: GitHubTokenUnavailableError, upstream: _Upstream, *, leaked: tuple[str, ...]
    ) -> None:
        message = str(error)
        assert message.startswith("github app token mint failed: ")
        assert _bearer_jwt(upstream.requests[0]) not in message
        for secret in leaked:
            assert secret not in message
        assert error.__cause__ is None  # no chained exception carrying the original text

    @pytest.mark.parametrize("status", [200, 401, 403, 404, 422, 500, 502])
    async def test_non_201_status(self, private_pem: bytes, status: int) -> None:
        body = "LEAKY-RESPONSE-BODY ghs_should_never_appear"
        upstream = _Upstream(lambda _r: httpx.Response(status, text=body))
        provider = _provider(private_pem, upstream)

        with pytest.raises(GitHubTokenUnavailableError) as excinfo:
            await provider.get_token()

        assert str(excinfo.value) == f"github app token mint failed: status {status}"
        self._assert_message_is_clean(excinfo.value, upstream, leaked=("LEAKY", "ghs_should"))

    async def test_timeout(self, private_pem: bytes) -> None:
        def raise_timeout(_request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectTimeout("leaky-transport-detail")

        upstream = _Upstream(raise_timeout)
        provider = _provider(private_pem, upstream)

        with pytest.raises(GitHubTokenUnavailableError) as excinfo:
            await provider.get_token()

        assert str(excinfo.value) == "github app token mint failed: ConnectTimeout"
        self._assert_message_is_clean(excinfo.value, upstream, leaked=("leaky-transport-detail",))

    @pytest.mark.parametrize(
        "response",
        [
            httpx.Response(201, text="not json at all LEAKY"),
            httpx.Response(201, json=["LEAKY"]),
            httpx.Response(201, json={"expires_at": "2026-01-01T01:00:00Z"}),
            httpx.Response(201, json={"token": "ghs_x", "expires_at": "LEAKY-not-a-date"}),
            httpx.Response(201, json={"token": "ghs_x"}),
            httpx.Response(201, json={"token": "", "expires_at": "2026-01-01T01:00:00Z"}),
            httpx.Response(201, json={"token": 12345, "expires_at": "2026-01-01T01:00:00Z"}),
            httpx.Response(201, json={"token": "ghs_x", "expires_at": "2026-01-01T01:00:00"}),
        ],
        ids=[
            "not-json",
            "json-list",
            "no-token",
            "bad-expiry",
            "no-expiry",
            "empty-token",
            "token-not-str",
            "naive-expiry",
        ],
    )
    async def test_malformed_body(self, private_pem: bytes, response: httpx.Response) -> None:
        upstream = _Upstream(lambda _r: response)
        provider = _provider(private_pem, upstream)

        with pytest.raises(GitHubTokenUnavailableError) as excinfo:
            await provider.get_token()

        self._assert_message_is_clean(excinfo.value, upstream, leaked=("LEAKY", "ghs_x"))

    async def test_unusable_private_key_is_unavailable_not_a_crash(self) -> None:
        upstream = _Upstream(lambda _r: _created())
        provider = _provider(b"-----BEGIN NOT A KEY-----", upstream)

        with pytest.raises(GitHubTokenUnavailableError, match="github app token mint failed"):
            await provider.get_token()

        assert upstream.requests == []  # never reached GitHub with a bad JWT

    async def test_failure_is_not_cached(self, private_pem: bytes) -> None:
        responses = iter([httpx.Response(502), _created("ghs_minted_2")])
        upstream = _Upstream(lambda _r: next(responses))
        provider = _provider(private_pem, upstream)

        with pytest.raises(GitHubTokenUnavailableError):
            await provider.get_token()
        token = await provider.get_token()

        assert token.value == "ghs_minted_2"
        assert len(upstream.requests) == 2


class TestTokenHygiene:
    """A8 — the value never reaches a repr (and so never a log line)."""

    def test_repr_hides_the_value(self) -> None:
        token = GitHubToken(value="ghs_super_secret_value", expires_at=_NOW)

        assert "ghs_super_secret_value" not in repr(token)
        assert "ghs_super_secret_value" not in str(token)
        assert "expires_at" in repr(token)

    async def test_provider_repr_hides_key_and_token(self, private_pem: bytes) -> None:
        upstream = _Upstream(lambda _r: _created("ghs_minted_1"))
        provider = _provider(private_pem, upstream)
        await provider.get_token()

        assert "ghs_minted_1" not in repr(provider)
        assert "PRIVATE KEY" not in repr(provider)


class TestStaticProvider:
    async def test_serves_the_pat_with_no_expiry(self) -> None:
        provider = StaticGitHubTokenProvider("ghp_static_pat")

        token = await provider.get_token()

        assert provider.mode is GitHubAuthMode.pat
        assert token.value == "ghp_static_pat"
        assert token.expires_at is None
        assert await provider.get_token() is token

    def test_rejects_an_empty_token(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            StaticGitHubTokenProvider("")


class TestBuildProvider:
    async def test_pat_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_github_env(monkeypatch)
        monkeypatch.setenv("AI_BUNDLES_GITHUB_TOKEN", "ghp_static_pat")
        settings = hermetic_settings(jwt_secret_key=_JWT_SECRET)

        provider = build_github_token_provider(settings, http_client=httpx.AsyncClient())

        assert isinstance(provider, StaticGitHubTokenProvider)
        assert (await provider.get_token()).value == "ghp_static_pat"

    def test_none_mode_builds_no_provider(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_github_env(monkeypatch)
        settings = hermetic_settings(jwt_secret_key=_JWT_SECRET)

        assert build_github_token_provider(settings, http_client=httpx.AsyncClient()) is None

    async def test_app_mode_mints_with_the_configured_ids_and_repos(
        self, monkeypatch: pytest.MonkeyPatch, private_pem: bytes
    ) -> None:
        import base64

        _clear_github_env(monkeypatch)
        monkeypatch.setenv("GITHUB_APP_ID", "12345")
        monkeypatch.setenv("GITHUB_APP_INSTALLATION_ID", "678")
        monkeypatch.setenv("GITHUB_APP_PRIVATE_KEY_B64", base64.b64encode(private_pem).decode())
        settings = hermetic_settings(jwt_secret_key=_JWT_SECRET)
        upstream = _Upstream(lambda _r: _created("ghs_minted_1"))

        provider = build_github_token_provider(settings, http_client=upstream.client())

        assert isinstance(provider, GitHubAppTokenProvider)
        assert provider.mode is GitHubAuthMode.app
        token = await provider.get_token()
        assert token.value == "ghs_minted_1"
        assert str(upstream.requests[0].url) == _MINT_URL
        assert json.loads(upstream.requests[0].content)["repositories"] == [
            "aisha",
            "ai-bundles",
        ]


def _clear_github_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "GITHUB_APP_ID",
        "GITHUB_APP_INSTALLATION_ID",
        "GITHUB_APP_PRIVATE_KEY_B64",
        "AI_BUNDLES_GITHUB_TOKEN",
    ):
        monkeypatch.delenv(name, raising=False)
