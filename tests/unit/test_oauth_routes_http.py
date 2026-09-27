"""HTTP-level tests for /v1/auth/oauth/* over a real Litestar app.

The real ``OAuthController`` + ``OAuthService`` run against an in-memory
flow store, a fake provider client, and mocked repositories/AuthService —
the DB-backed behaviours are covered in tests/integration/test_oauth_flow.py.
Contracts: I3 (redirect_uri not from Host), I4 (route mapping), I13
(commit before handoff), I18 (404 when disabled), I21 (return_to in the
fragment), I22 (no secrets in logs), plus error/cookie mapping.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
import structlog
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from litestar.testing import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.app import legal_submission_incomplete_handler, legal_version_stale_handler
from src.api.dependencies.common import get_product_config, get_product_id
from src.api.middleware.product import ProductMiddleware
from src.api.routes.oauth import OAuthController
from src.api.security.jwt import JWTConfig, JWTService
from src.api.security.oauth_tx_cookie import OAUTH_TX_COOKIE
from src.api.services.auth import EmailAlreadyExistsError, TokenPair
from src.api.services.legal.acceptance import RequestContext
from src.api.services.legal.errors import (
    LegalSubmissionIncompleteError,
    LegalVersionStaleError,
)
from src.api.services.oauth.errors import EmailUnverifiedError, OAuthFailedError
from src.api.services.oauth.models import OAuthFlow
from src.api.services.oauth.service import OAuthService
from src.api.services.token_revocation import TokenRevocationService
from src.core.product import OAuthProvider
from src.core.product_registry import VEX_CONFIG
from tests.oauth_support import (
    API_PUBLIC_URL,
    APP_URL,
    FakeProviderClient,
    InMemoryOAuthFlowStore,
    fake_registry,
    fragment_params,
    identity,
    oauth_settings,
    query_params,
)

if TYPE_CHECKING:
    from src.api.services.oauth.registry import OAuthProviderRegistry
    from src.core.config import Settings

pytestmark = pytest.mark.unit

TEST_SECRET = "test_secret_key_for_testing_only_256bits_long"
VEX = {"X-Product-Id": "vex"}
SYNTHARA = {"X-Product-Id": "synthara"}
CALLBACK_PREFIX = f"{APP_URL}/auth/callback#"


def _nested() -> MagicMock:
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=None)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


@dataclass
class Harness:
    """One app + its collaborators, exposed for assertions."""

    settings: Settings = field(default_factory=oauth_settings)
    provider: FakeProviderClient = field(default_factory=lambda: FakeProviderClient(identity()))
    store: InMemoryOAuthFlowStore = field(default_factory=InMemoryOAuthFlowStore)
    registry: OAuthProviderRegistry | None = None
    order: MagicMock = field(default_factory=MagicMock)

    def __post_init__(self) -> None:
        if self.registry is None:
            self.registry = fake_registry(self.settings, self.provider)
        self.jwt = JWTService(JWTConfig(secret_key=TEST_SECRET))
        self.session = MagicMock(spec=AsyncSession)
        self.session.commit = AsyncMock()
        self.session.rollback = AsyncMock()
        self.session.flush = AsyncMock()
        self.session.begin_nested = MagicMock(side_effect=_nested)
        self.identity_repo = MagicMock()
        self.identity_repo.get_by_subject = AsyncMock(return_value=None)
        self.identity_repo.get_for_user = AsyncMock(return_value=None)
        self.identity_repo.touch_last_login = AsyncMock()
        self.user_repo = MagicMock()
        self.user_repo.get_active_user_by_email = AsyncMock(return_value=None)
        self.user_repo.get_user = AsyncMock(return_value=None)
        self.user = MagicMock(id=uuid4(), is_active=True)
        self.user_repo.get_active_user = AsyncMock(return_value=self.user)
        self.auth = MagicMock()
        self.auth.validate_signup = MagicMock(return_value=MagicMock(name="validated"))
        self.auth.provision_user = AsyncMock(return_value=self.user)
        self.tokens = TokenPair(
            access_token="access-token-value",
            refresh_token="refresh-token-value",
            expires_at=datetime.now(UTC) + timedelta(minutes=15),
            expires_in=900,
        )
        self.auth.issue_session = AsyncMock(return_value=self.tokens)
        # I13 spy: commit vs. the Redis handoff write.
        self.order.attach_mock(self.session.commit, "commit")
        put_handoff = self.store.put_handoff
        put_signup = self.store.put_signup

        async def spy_handoff(*args: Any) -> None:
            self.order.put_handoff()
            await put_handoff(*args)

        async def spy_signup(*args: Any) -> None:
            self.order.put_signup()
            await put_signup(*args)

        self.store.put_handoff = spy_handoff  # type: ignore[method-assign]
        self.store.put_signup = spy_signup  # type: ignore[method-assign]
        assert self.registry is not None
        self.service = OAuthService(
            registry=self.registry,
            store=self.store,
            identity_repo=self.identity_repo,
            user_repo=self.user_repo,
            auth_service=self.auth,
            session=self.session,
            settings=self.settings,
        )

    def app(self) -> Litestar:
        return Litestar(
            route_handlers=[OAuthController],
            middleware=[ProductMiddleware],
            dependencies={
                "product_config": Provide(get_product_config, sync_to_thread=False),
                "product_id": Provide(get_product_id, sync_to_thread=False),
                "oauth_service": Provide(lambda: self.service, sync_to_thread=False),
                "session": Provide(lambda: self.session, sync_to_thread=False),
                "jwt_service": Provide(lambda: self.jwt, sync_to_thread=False),
                "settings": Provide(lambda: self.settings, sync_to_thread=False),
                "request_context": Provide(
                    lambda: RequestContext(ip_address="203.0.113.9", user_agent="ua"),
                    sync_to_thread=False,
                ),
            },
            exception_handlers={
                LegalSubmissionIncompleteError: legal_submission_incomplete_handler,
                LegalVersionStaleError: legal_version_stale_handler,
            },
            state=State(
                {
                    "jwt_service": self.jwt,
                    "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                }
            ),
        )


def _authorize(client: TestClient[Litestar], **params: str) -> Any:
    return client.get(
        "/v1/auth/oauth/google/authorize",
        params=params,
        headers=VEX,
        follow_redirects=False,
    )


def _callback(
    client: TestClient[Litestar], headers: dict[str, str] | None = None, **params: str
) -> Any:
    return client.get(
        "/v1/auth/oauth/google/callback",
        params=params,
        headers=headers or VEX,
        follow_redirects=False,
    )


def _start(client: TestClient[Litestar], **params: str) -> str:
    """Authorize and return the ``state`` sent to the provider."""
    resp = _authorize(client, **params)
    assert resp.status_code == 302
    return query_params(resp.headers["location"])["state"]


# ---------------------------------------------------------------------------
# authorize
# ---------------------------------------------------------------------------


class TestAuthorize:
    def test_redirects_to_provider_with_binding_cookie(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            resp = _authorize(client, return_to="/library")
        assert resp.status_code == 302
        assert resp.headers["location"].startswith("https://provider.test/auth?")
        assert resp.headers["cache-control"] == "no-store"
        set_cookie = resp.headers["set-cookie"]
        assert set_cookie.startswith(f"{OAUTH_TX_COOKIE}=")
        assert "HttpOnly" in set_cookie
        assert "Path=/v1/auth/oauth" in set_cookie
        assert "SameSite=lax" in set_cookie or "SameSite=Lax" in set_cookie
        assert "Domain" not in set_cookie
        assert f"Max-Age={h.settings.oauth_signup_ticket_ttl_seconds}" in set_cookie
        (flow,) = h.store.flows.values()
        assert flow.return_to == "/library"
        assert flow.product_id == "vex"
        assert h.store.ttls[next(iter(h.store.ttls))] == h.settings.oauth_flow_ttl_seconds

    def test_redirect_uri_ignores_spoofed_host(self) -> None:
        """I3 — redirect_uri comes from API_PUBLIC_URL_VEX, never the Host header."""
        h = Harness()
        with TestClient(app=h.app()) as client:
            resp = client.get(
                "/v1/auth/oauth/google/authorize",
                headers={**VEX, "Host": "evil.example", "X-Forwarded-Host": "evil.example"},
                follow_redirects=False,
            )
        params = query_params(resp.headers["location"])
        assert params["redirect_uri"] == f"{API_PUBLIC_URL}/v1/auth/oauth/google/callback"

    def test_challenge_matches_stored_verifier(self) -> None:
        from src.api.services.oauth.pkce import s256_challenge

        h = Harness()
        with TestClient(app=h.app()) as client:
            resp = _authorize(client)
        params = query_params(resp.headers["location"])
        flow = h.store.flows[params["state"]]
        assert params["code_challenge"] == s256_challenge(flow.code_verifier)
        assert params["nonce"] == flow.nonce

    def test_unknown_provider_404(self) -> None:
        with TestClient(app=Harness().app()) as client:
            resp = client.get("/v1/auth/oauth/myspace/authorize", headers=VEX)
        assert resp.status_code == 404

    def test_disabled_for_product_404(self) -> None:
        """I18 — allowed-but-unconfigured (synthara here) is a 404, no flow stored."""
        h = Harness()
        with TestClient(app=h.app()) as client:
            resp = client.get(
                "/v1/auth/oauth/google/authorize", headers=SYNTHARA, follow_redirects=False
            )
        assert resp.status_code == 404
        assert h.store.flows == {}
        assert "set-cookie" not in resp.headers

    @pytest.mark.parametrize("value", ["//evil.com", "https://evil.com", "/a b"])
    def test_bad_return_to_400(self, value: str) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            resp = _authorize(client, return_to=value)
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_return_to"
        assert h.store.flows == {}


# ---------------------------------------------------------------------------
# callback
# ---------------------------------------------------------------------------


class TestCallback:
    def test_signup_outcome(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client, return_to="/library?tab=fav&x=1")
            resp = _callback(client, code="provider-code", state=state)
        assert resp.status_code == 302
        location = resp.headers["location"]
        assert location.startswith(CALLBACK_PREFIX)
        frag = fragment_params(location)
        assert frag["result"] == "signup"
        assert frag["return_to"] == "/library?tab=fav&x=1"  # I21 round-trip
        assert "%2Flibrary%3Ftab%3Dfav%26x%3D1" in location  # URL-encoded
        pending = h.store.signups[frag["ticket"]]
        assert pending.email == "person@example.com"
        assert h.store.flows == {}  # state consumed
        assert resp.headers["referrer-policy"] == "no-referrer"

    def test_login_outcome_commits_before_handoff(self) -> None:
        """I13 — DB writes are committed before the handoff lands in Redis."""
        h = Harness()
        linked = MagicMock(id=uuid4(), user_id=h.user.id)
        h.identity_repo.get_by_subject = AsyncMock(return_value=linked)
        h.user_repo.get_user = AsyncMock(return_value=h.user)
        with TestClient(app=h.app()) as client:
            state = _start(client)
            resp = _callback(client, code="provider-code", state=state)
        frag = fragment_params(resp.headers["location"])
        assert frag["result"] == "login"
        assert "return_to" not in frag
        assert h.store.handoffs[frag["code"]].user_id == h.user.id
        assert [c[0] for c in h.order.mock_calls] == ["commit", "put_handoff"]
        h.identity_repo.touch_last_login.assert_awaited_once_with(linked.id)

    def test_provider_received_stored_verifier_and_redirect_uri(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client)
            flow = h.store.flows[state]
            _callback(client, code="provider-code", state=state)
        (exchange,) = h.provider.exchanges
        assert exchange == {
            "code": "provider-code",
            "code_verifier": flow.code_verifier,
            "redirect_uri": f"{API_PUBLIC_URL}/v1/auth/oauth/google/callback",
            "nonce": flow.nonce,
        }

    @pytest.mark.parametrize(
        ("error", "expected"),
        [("access_denied", "oauth_cancelled"), ("server_error", "oauth_failed")],
    )
    def test_provider_error_param(self, error: str, expected: str) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client)
            resp = _callback(client, error=error, state=state)
        assert fragment_params(resp.headers["location"]) == {"result": "error", "error": expected}
        assert h.provider.exchanges == []

    @pytest.mark.parametrize("missing", ["code", "state"])
    def test_missing_code_or_state(self, missing: str) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client)
            params = {"code": "c", "state": state}
            params.pop(missing)
            resp = client.get(
                "/v1/auth/oauth/google/callback",
                params=params,
                headers=VEX,
                follow_redirects=False,
            )
        assert fragment_params(resp.headers["location"])["error"] == "flow_expired"

    def test_without_cookie(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client)
            client.cookies.clear()
            resp = _callback(client, code="c", state=state)
        assert fragment_params(resp.headers["location"])["error"] == "flow_expired"
        assert h.provider.exchanges == []

    def test_with_other_cookie(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client)
            client.cookies.clear()
            client.cookies.set(OAUTH_TX_COOKIE, "forged-binding-value")
            resp = _callback(client, code="c", state=state)
        assert fragment_params(resp.headers["location"])["error"] == "flow_expired"

    def test_replayed_state(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client)
            _callback(client, code="c", state=state)
            resp = _callback(client, code="c", state=state)
        assert fragment_params(resp.headers["location"])["error"] == "flow_expired"
        assert len(h.provider.exchanges) == 1

    def test_product_mismatch(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            state = _start(client)
            resp = _callback(client, headers=SYNTHARA, code="c", state=state)
        location = resp.headers["location"]
        assert location.startswith(h.settings.app_url_synthara)
        assert fragment_params(location)["error"] == "flow_expired"

    @pytest.mark.parametrize(
        ("error", "expected"),
        [(OAuthFailedError(), "oauth_failed"), (EmailUnverifiedError(), "email_unverified")],
    )
    def test_provider_exchange_errors(self, error: Exception, expected: str) -> None:
        h = Harness(provider=FakeProviderClient(error=error))
        with TestClient(app=h.app()) as client:
            state = _start(client)
            resp = _callback(client, code="c", state=state)
        assert fragment_params(resp.headers["location"])["error"] == expected

    def test_unexpected_exception_is_still_a_redirect(self) -> None:
        h = Harness(provider=FakeProviderClient(error=RuntimeError("boom")))
        with TestClient(app=h.app()) as client:
            state = _start(client)
            resp = _callback(client, code="c", state=state)
        assert resp.status_code == 302
        assert fragment_params(resp.headers["location"])["error"] == "oauth_failed"
        h.session.rollback.assert_awaited_once()


class TestProviderMismatch:
    """I4 — a flow started for another provider is refused (service level; one enum member)."""

    async def test_provider_mismatch_is_flow_expired(self) -> None:
        from src.api.security.oauth_tx_cookie import binding_hash
        from src.api.services.oauth.errors import FlowExpiredError

        h = Harness()
        await h.store.put_flow(
            "st",
            OAuthFlow(
                product_id="vex",
                provider="apple",  # type: ignore[arg-type]
                code_verifier="v",
                nonce="n",
                binding_hash=binding_hash("b"),
                return_to=None,
            ),
            600,
        )
        with pytest.raises(FlowExpiredError):
            await h.service.resolve_callback(
                product=VEX_CONFIG, provider=OAuthProvider.GOOGLE, state="st", code="c", binding="b"
            )


# ---------------------------------------------------------------------------
# exchange / signup-info / complete-signup
# ---------------------------------------------------------------------------


def _login_code(h: Harness, client: TestClient[Litestar]) -> str:
    linked = MagicMock(id=uuid4(), user_id=h.user.id)
    h.identity_repo.get_by_subject = AsyncMock(return_value=linked)
    h.user_repo.get_user = AsyncMock(return_value=h.user)
    state = _start(client)
    return fragment_params(_callback(client, code="c", state=state).headers["location"])["code"]


def _signup_ticket(client: TestClient[Litestar]) -> str:
    state = _start(client)
    return fragment_params(_callback(client, code="c", state=state).headers["location"])["ticket"]


class TestExchange:
    def test_success_returns_tokens_and_clears_cookie(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            code = _login_code(h, client)
            resp = client.post("/v1/auth/oauth/exchange", json={"code": code}, headers=VEX)
        assert resp.status_code == 200
        body = resp.json()
        assert body["access_token"] == "access-token-value"
        assert body["refresh_token"] == "refresh-token-value"
        assert set(body) == {
            "access_token",
            "refresh_token",
            "token_type",
            "expires_in",
            "expires_at",
            "content_cookie_expires_at",
        }
        cookies = resp.headers.get_list("set-cookie")
        assert any(c.startswith("apex_content=") for c in cookies)
        assert any(c.startswith(f"{OAUTH_TX_COOKIE}=") and "Max-Age=0" in c for c in cookies)
        h.auth.issue_session.assert_awaited_once()

    def test_second_exchange_rejected(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            code = _login_code(h, client)
            client.post("/v1/auth/oauth/exchange", json={"code": code}, headers=VEX)
            resp = client.post("/v1/auth/oauth/exchange", json={"code": code}, headers=VEX)
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_handoff"

    def test_exchange_without_cookie_rejected(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            code = _login_code(h, client)
            client.cookies.clear()
            resp = client.post("/v1/auth/oauth/exchange", json={"code": code}, headers=VEX)
        assert resp.status_code == 400
        h.auth.issue_session.assert_not_awaited()

    def test_inactive_user_401(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            code = _login_code(h, client)
            h.user_repo.get_active_user = AsyncMock(return_value=None)
            resp = client.post("/v1/auth/oauth/exchange", json={"code": code}, headers=VEX)
        assert resp.status_code == 401
        assert resp.json()["error"] == "account_inactive"

    @pytest.mark.parametrize(
        "body", [{}, {"code": "short"}, {"code": "x" * 200}, {"code": "a" * 43, "extra": 1}]
    )
    def test_malformed_body_400(self, body: dict[str, Any]) -> None:
        with TestClient(app=Harness().app()) as client:
            resp = client.post("/v1/auth/oauth/exchange", json=body, headers=VEX)
        assert resp.status_code == 400


class TestSignup:
    def test_signup_info_is_repeatable(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            ticket = _signup_ticket(client)
            for _ in range(2):
                resp = client.post(
                    "/v1/auth/oauth/signup-info", json={"ticket": ticket}, headers=VEX
                )
                assert resp.status_code == 200
                assert resp.json() == {"email": "person@example.com", "provider": "google"}

    def test_signup_info_wrong_cookie(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            ticket = _signup_ticket(client)
            client.cookies.clear()
            client.cookies.set(OAUTH_TX_COOKIE, "forged-binding-value")
            resp = client.post("/v1/auth/oauth/signup-info", json={"ticket": ticket}, headers=VEX)
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_signup_ticket"

    def test_complete_signup_201(self) -> None:
        h = Harness()
        with TestClient(app=h.app()) as client:
            ticket = _signup_ticket(client)
            resp = client.post(
                "/v1/auth/oauth/complete-signup",
                json={"ticket": ticket, "accepted_documents": [], "display_name": "Pat"},
                headers=VEX,
            )
        assert resp.status_code == 201
        assert resp.json()["access_token"] == "access-token-value"
        cookies = resp.headers.get_list("set-cookie")
        assert any(c.startswith(f"{OAUTH_TX_COOKIE}=") and "Max-Age=0" in c for c in cookies)
        kwargs = h.auth.provision_user.await_args.kwargs
        assert kwargs["password_hash"] is None
        assert kwargs["email"] == "person@example.com"
        assert kwargs["display_name"] == "Pat"
        assert kwargs["email_verified_at"] is not None
        assert h.store.signups == {}

    def test_stale_legal_409_keeps_ticket(self) -> None:
        h = Harness()
        h.auth.validate_signup = MagicMock(side_effect=LegalVersionStaleError([]))
        with TestClient(app=h.app()) as client:
            ticket = _signup_ticket(client)
            resp = client.post(
                "/v1/auth/oauth/complete-signup",
                json={"ticket": ticket, "accepted_documents": []},
                headers=VEX,
            )
        assert resp.status_code == 409
        assert resp.json()["error"] == "legal_version_stale"
        assert ticket in h.store.signups
        h.auth.provision_user.assert_not_awaited()

    def test_email_exists_400(self) -> None:
        h = Harness()
        h.auth.provision_user = AsyncMock(side_effect=EmailAlreadyExistsError("taken"))
        with TestClient(app=h.app()) as client:
            ticket = _signup_ticket(client)
            resp = client.post(
                "/v1/auth/oauth/complete-signup",
                json={"ticket": ticket, "accepted_documents": []},
                headers=VEX,
            )
        assert resp.status_code == 400
        assert resp.json()["error"] == "email_exists"

    def test_identity_conflict_409(self) -> None:
        from sqlalchemy.exc import IntegrityError

        h = Harness()
        orig = Exception(
            "duplicate key value violates unique constraint "
            '"uq_user_identities_product_provider_subject"'
        )

        def savepoint() -> MagicMock:
            # Releasing a savepoint flushes: a clean exit raises the violation,
            # an exceptional exit (the outer savepoint) just propagates.
            cm = MagicMock()
            cm.__aenter__ = AsyncMock(return_value=None)

            async def aexit(exc_type: Any, *_: Any) -> bool:
                if exc_type is None:
                    raise IntegrityError("INSERT", {}, orig)
                return False

            cm.__aexit__ = AsyncMock(side_effect=aexit)
            return cm

        h.session.begin_nested = MagicMock(side_effect=savepoint)
        with TestClient(app=h.app()) as client:
            ticket = _signup_ticket(client)
            resp = client.post(
                "/v1/auth/oauth/complete-signup",
                json={"ticket": ticket, "accepted_documents": []},
                headers=VEX,
            )
        assert resp.status_code == 409
        assert resp.json()["error"] == "identity_conflict"
        h.auth.issue_session.assert_not_awaited()

    def test_display_name_bounds(self) -> None:
        with TestClient(app=Harness().app()) as client:
            resp = client.post(
                "/v1/auth/oauth/complete-signup",
                json={"ticket": "a" * 43, "accepted_documents": [], "display_name": "x" * 101},
                headers=VEX,
            )
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# I22 — no secrets in logs
# ---------------------------------------------------------------------------


class TestNoSecretsLogged:
    def test_full_flow_logs_no_secret_values(self) -> None:
        h = Harness()
        secrets: set[str] = {"access-token-value", "refresh-token-value", "provider-code"}
        secrets |= {"google-subject-123", "person@example.com"}
        with structlog.testing.capture_logs() as logs, TestClient(app=h.app()) as client:
            # signup path
            resp = _authorize(client)
            state = query_params(resp.headers["location"])["state"]
            flow = h.store.flows[state]
            secrets |= {state, flow.code_verifier, flow.nonce, client.cookies[OAUTH_TX_COOKIE]}
            cb = _callback(client, code="provider-code", state=state)
            ticket = fragment_params(cb.headers["location"])["ticket"]
            secrets.add(ticket)
            client.post("/v1/auth/oauth/signup-info", json={"ticket": ticket}, headers=VEX)
            client.post(
                "/v1/auth/oauth/complete-signup",
                json={"ticket": ticket, "accepted_documents": []},
                headers=VEX,
            )
            # login path (+ rejected replay)
            code = _login_code(h, client)
            secrets |= {code, client.cookies[OAUTH_TX_COOKIE]}
            client.post("/v1/auth/oauth/exchange", json={"code": code}, headers=VEX)
            client.post("/v1/auth/oauth/exchange", json={"code": code}, headers=VEX)
            _callback(client, code="provider-code", state=state)  # replay → rejected

        assert logs, "expected the flow to log"
        rendered = json.dumps(logs, default=str)
        leaked = sorted(s for s in secrets if s in rendered)
        assert leaked == []
