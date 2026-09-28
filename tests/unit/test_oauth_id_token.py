"""JwksIdTokenVerifier — I1 (claim/signature rejection) and I2 (async JWKS cache/refetch)."""

from __future__ import annotations

import ast
import inspect
import time
from typing import Any

import httpx
import pytest

from src.api.services.oauth import id_token as id_token_module
from src.api.services.oauth.errors import EmailUnverifiedError, OAuthFailedError
from src.api.services.oauth.id_token import JwksIdTokenVerifier
from src.api.services.oauth.providers.google import ISSUERS, JWKS_URI
from src.core.product import OAuthProvider
from tests.oauth_support import CLIENT_ID, RsaSigner, jwks

pytestmark = pytest.mark.unit

NONCE = "expected-nonce"


class JwksServer:
    """httpx MockTransport serving a mutable JWKS and counting fetches."""

    def __init__(self, *signers: RsaSigner) -> None:
        self.document: dict[str, Any] = jwks(*signers)
        self.fetches = 0
        self.status_code = 200

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == JWKS_URI
        self.fetches += 1
        return httpx.Response(self.status_code, json=self.document)

    def client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(transport=httpx.MockTransport(self.handler))


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _verifier(server: JwksServer, clock: Clock | None = None) -> JwksIdTokenVerifier:
    return JwksIdTokenVerifier(
        server.client(),
        provider=OAuthProvider.GOOGLE,
        jwks_uri=JWKS_URI,
        issuers=ISSUERS,
        cache_ttl_seconds=3600,
        clock=clock or Clock(),
    )


class TestAccepts:
    async def test_valid_token_returns_lowercased_identity(self) -> None:
        signer = RsaSigner()
        identity = await _verifier(JwksServer(signer)).verify(
            signer.token(), audience=CLIENT_ID, nonce=NONCE
        )
        assert identity.provider is OAuthProvider.GOOGLE
        assert identity.subject == "google-subject-123"
        assert identity.email == "person@example.com"

    @pytest.mark.parametrize("issuer", sorted(ISSUERS))
    async def test_both_google_issuer_forms_accepted(self, issuer: str) -> None:
        signer = RsaSigner()
        await _verifier(JwksServer(signer)).verify(
            signer.token(iss=issuer), audience=CLIENT_ID, nonce=NONCE
        )

    async def test_azp_absent_is_accepted(self) -> None:
        signer = RsaSigner()
        await _verifier(JwksServer(signer)).verify(
            signer.token(azp=None), audience=CLIENT_ID, nonce=NONCE
        )


class TestRejects:
    """I1 — every listed defect is refused with the right error type."""

    @pytest.mark.parametrize(
        ("claims", "error"),
        [
            ({"aud": "someone-else"}, OAuthFailedError),
            ({"iss": "https://evil.example"}, OAuthFailedError),
            ({"exp": int(time.time()) - 3600, "iat": int(time.time()) - 7200}, OAuthFailedError),
            ({"nonce": "other-nonce"}, OAuthFailedError),
            ({"nonce": None}, OAuthFailedError),
            ({"azp": "someone-else"}, OAuthFailedError),
            ({"email_verified": None}, EmailUnverifiedError),
            ({"email_verified": False}, EmailUnverifiedError),
            ({"email_verified": "true"}, OAuthFailedError),
            ({"email": None}, OAuthFailedError),
            ({"sub": None}, OAuthFailedError),
        ],
        ids=[
            "wrong_aud",
            "wrong_iss",
            "expired",
            "nonce_mismatch",
            "nonce_missing",
            "azp_mismatch",
            "email_verified_missing",
            "email_verified_false",
            "email_verified_string",
            "email_missing",
            "sub_missing",
        ],
    )
    async def test_claim_defects(self, claims: dict[str, Any], error: type[Exception]) -> None:
        signer = RsaSigner()
        with pytest.raises(error):
            await _verifier(JwksServer(signer)).verify(
                signer.token(**claims), audience=CLIENT_ID, nonce=NONCE
            )

    async def test_email_unverified_is_not_an_oauth_failed(self) -> None:
        assert not issubclass(EmailUnverifiedError, OAuthFailedError)

    async def test_bad_signature(self) -> None:
        published, forger = RsaSigner("key-1"), RsaSigner("key-1")
        with pytest.raises(OAuthFailedError):
            await _verifier(JwksServer(published)).verify(
                forger.token(), audience=CLIENT_ID, nonce=NONCE
            )

    async def test_hs256_token_rejected(self) -> None:
        import jwt

        token = jwt.encode(
            {"sub": "x"}, "secret-key-at-least-32-bytes-long!!", headers={"kid": "key-1"}
        )
        with pytest.raises(OAuthFailedError):
            await _verifier(JwksServer(RsaSigner())).verify(token, audience=CLIENT_ID, nonce=NONCE)

    @pytest.mark.parametrize("token", ["not-a-jwt", "a.b.c"])
    async def test_malformed_token(self, token: str) -> None:
        with pytest.raises(OAuthFailedError):
            await _verifier(JwksServer(RsaSigner())).verify(token, audience=CLIENT_ID, nonce=NONCE)

    async def test_jwks_endpoint_failure(self) -> None:
        signer = RsaSigner()
        server = JwksServer(signer)
        server.status_code = 503
        with pytest.raises(OAuthFailedError):
            await _verifier(server).verify(signer.token(), audience=CLIENT_ID, nonce=NONCE)


class TestJwksCache:
    """I2 — cached by kid; unknown kid → exactly one throttled forced refetch."""

    async def test_keys_cached_across_verifications(self) -> None:
        signer = RsaSigner()
        server = JwksServer(signer)
        verifier = _verifier(server)
        for _ in range(3):
            await verifier.verify(signer.token(), audience=CLIENT_ID, nonce=NONCE)
        assert server.fetches == 1

    async def test_ttl_expiry_refetches(self) -> None:
        signer = RsaSigner()
        server = JwksServer(signer)
        clock = Clock()
        verifier = _verifier(server, clock)
        await verifier.verify(signer.token(), audience=CLIENT_ID, nonce=NONCE)
        clock.now += 3600
        await verifier.verify(signer.token(), audience=CLIENT_ID, nonce=NONCE)
        assert server.fetches == 2

    async def test_unknown_kid_triggers_exactly_one_refetch_then_succeeds(self) -> None:
        old, rotated = RsaSigner("old"), RsaSigner("rotated")
        server = JwksServer(old)
        verifier = _verifier(server)
        await verifier.verify(old.token(), audience=CLIENT_ID, nonce=NONCE)
        server.document = jwks(old, rotated)

        await verifier.verify(rotated.token(), audience=CLIENT_ID, nonce=NONCE)
        assert server.fetches == 2

    async def test_still_unknown_kid_fails_after_one_refetch(self) -> None:
        signer = RsaSigner()
        server = JwksServer(signer)
        verifier = _verifier(server)
        await verifier.verify(signer.token(), audience=CLIENT_ID, nonce=NONCE)

        with pytest.raises(OAuthFailedError):
            await verifier.verify(signer.token(_kid="bogus"), audience=CLIENT_ID, nonce=NONCE)
        assert server.fetches == 2

    async def test_forced_refetch_is_throttled(self) -> None:
        signer = RsaSigner()
        server = JwksServer(signer)
        clock = Clock()
        verifier = _verifier(server, clock)
        await verifier.verify(signer.token(), audience=CLIENT_ID, nonce=NONCE)

        for i in range(5):
            with pytest.raises(OAuthFailedError):
                await verifier.verify(
                    signer.token(_kid=f"bogus-{i}"), audience=CLIENT_ID, nonce=NONCE
                )
        assert server.fetches == 2  # initial + one forced

        clock.now += 61
        with pytest.raises(OAuthFailedError):
            await verifier.verify(signer.token(_kid="bogus-late"), audience=CLIENT_ID, nonce=NONCE)
        assert server.fetches == 3

    async def test_cold_cache_unknown_kid_fetches_once(self) -> None:
        signer = RsaSigner()
        server = JwksServer(signer)
        with pytest.raises(OAuthFailedError):
            await _verifier(server).verify(
                signer.token(_kid="bogus"), audience=CLIENT_ID, nonce=NONCE
            )
        assert server.fetches == 1

    async def test_concurrent_unknown_kids_share_one_refetch(self) -> None:
        import asyncio

        signer = RsaSigner()
        server = JwksServer(signer)
        verifier = _verifier(server)
        await verifier.verify(signer.token(), audience=CLIENT_ID, nonce=NONCE)

        results = await asyncio.gather(
            *(
                verifier.verify(signer.token(_kid=f"x{i}"), audience=CLIENT_ID, nonce=NONCE)
                for i in range(10)
            ),
            return_exceptions=True,
        )
        assert all(isinstance(r, OAuthFailedError) for r in results)
        assert server.fetches == 2

    def test_never_uses_blocking_pyjwkclient(self) -> None:
        tree = ast.parse(inspect.getsource(id_token_module))
        names = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)} | {
            n.id for n in ast.walk(tree) if isinstance(n, ast.Name)
        }
        assert "PyJWKClient" not in names
