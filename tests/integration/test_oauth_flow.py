"""OAuth sign-in against real PostgreSQL 16 (and real Redis when REDIS_URL is set).

Drives the real ``OAuthController`` + ``OAuthService`` + ``AuthService`` over
the per-test savepoint session; only the provider (Google) is faked. The
flow store is in-memory by default and ``RedisOAuthFlowStore`` against a real
Redis for the single-use/concurrency contracts.

Contracts: I4 (callback rejections), I5 (single use + concurrent signup),
I6 (binding cookie), I7 (no rows before consent), I8 (stale legal keeps the
ticket), I9 (signup rows), I10 (auto-link rules), I11 (admin-deactivated),
I12 (self-closure unlinks), I14 (lgl digest), I17 (reset verifies email),
I23 (migration 048).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import jwt as pyjwt
import msgspec
import pytest
import pytest_asyncio
import redis.asyncio as aioredis
import structlog
from alembic.operations import Operations
from alembic.runtime.migration import MigrationContext
from litestar import Litestar
from litestar.datastructures import State
from litestar.di import Provide
from sqlalchemy import delete, func, select, text

from src.api.app import (
    legal_account_inactive_handler,
    legal_submission_incomplete_handler,
    legal_version_stale_handler,
)
from src.api.dependencies.common import get_product_config, get_product_id
from src.api.middleware.product import ProductMiddleware
from src.api.routes.oauth import OAuthController
from src.api.schemas.ops_events import OpsEventType
from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.security.guards import _enforce_legal_acceptance
from src.api.security.oauth_tx_cookie import OAUTH_TX_COOKIE, binding_hash
from src.api.services.auth import AuthService
from src.api.services.email_verification import EmailVerificationService
from src.api.services.legal.acceptance import LegalAcceptanceService, RequestContext
from src.api.services.legal.errors import (
    LegalAccountInactiveError,
    LegalSubmissionIncompleteError,
    LegalVersionStaleError,
)
from src.api.services.oauth.errors import InvalidSignupTicketError
from src.api.services.oauth.flow_store import RedisOAuthFlowStore
from src.api.services.oauth.models import PendingSignup
from src.api.services.oauth.service import OAuthService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService
from src.core.enums import LegalAcceptanceSource
from src.core.product import OAuthProvider
from src.core.product_registry import VEX_CONFIG
from src.core.uid import new_id
from src.db.models.billing import TokenAccount
from src.db.models.legal import LegalAcceptance
from src.db.models.user import User
from src.db.models.user_identity import UserIdentity
from src.db.repositories.legal import LegalAcceptanceRepository
from src.db.repositories.user import UserRepository
from src.db.repositories.user_identity import UserIdentityRepository
from tests.legal_support import accept_all_current, make_legal_registry
from tests.oauth_support import (
    FakeProviderClient,
    InMemoryOAuthFlowStore,
    fake_registry,
    fragment_params,
    identity,
    oauth_settings,
    query_params,
)

if TYPE_CHECKING:
    import contextlib
    from collections.abc import AsyncGenerator
    from types import ModuleType

    from sqlalchemy import Connection
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from src.api.services.legal.registry import LegalDocumentRegistry
    from src.api.services.oauth.flow_store import OAuthFlowStore
    from tests.integration.conftest import ResetTokenFactory, UserFactory

JWT_SECRET = "integration-oauth-secret-key-32-bytes-long"
CONTEXT = RequestContext(ip_address="192.0.2.55", user_agent="IntegrationOAuth/1.0")
VEX = {"X-Product-Id": "vex"}
SUBJECT = "google-sub-integration"
_REDIS_URL = os.environ.get("REDIS_URL")


def _today() -> date:
    return datetime.now(UTC).date()


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class Flow:
    """One app over the test session, with helpers for each step."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        store: OAuthFlowStore | None = None,
        email: str = "person@example.com",
        subject: str = SUBJECT,
    ) -> None:
        self.session = session
        self.settings = oauth_settings()
        self.provider = FakeProviderClient(identity(subject=subject, email=email))
        self.store = store if store is not None else InMemoryOAuthFlowStore()
        self.legal_registry = make_legal_registry()
        self.jwt = JWTService(JWTConfig(secret_key=JWT_SECRET))
        self.ops = MagicMock()
        self.ops.publish = AsyncMock()
        self.service = build_service(
            session,
            store=self.store,
            provider=self.provider,
            legal_registry=self.legal_registry,
            jwt_service=self.jwt,
            ops=self.ops,
        )

    def client(self) -> contextlib.AbstractAsyncContextManager[httpx.AsyncClient]:
        """In-loop ASGI client (the app shares the test's asyncpg connection)."""
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=cast("Any", self.app())),
            base_url="http://testserver.local",
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
                "request_context": Provide(lambda: CONTEXT, sync_to_thread=False),
            },
            exception_handlers={
                LegalSubmissionIncompleteError: legal_submission_incomplete_handler,
                LegalVersionStaleError: legal_version_stale_handler,
                LegalAccountInactiveError: legal_account_inactive_handler,
            },
            state=State(
                {
                    "jwt_service": self.jwt,
                    "token_revocation": TokenRevocationService(None, max_token_ttl_seconds=0),
                    "legal_registry": self.legal_registry,
                }
            ),
        )

    async def callback(self, client: httpx.AsyncClient, **headers: str) -> dict[str, str]:
        """authorize → callback; returns the fragment parameters."""
        resp = await client.get(
            "/v1/auth/oauth/google/authorize", headers=VEX, follow_redirects=False
        )
        assert resp.status_code == 302
        state = query_params(resp.headers["location"])["state"]
        cb = await client.get(
            "/v1/auth/oauth/google/callback",
            params={"code": "provider-code", "state": state},
            headers={**VEX, **headers},
            follow_redirects=False,
        )
        assert cb.status_code == 302
        return fragment_params(cb.headers["location"])

    async def complete(
        self, client: httpx.AsyncClient, ticket: str, documents: list[Any] | None = None
    ) -> Any:
        submission = (
            documents
            if documents is not None
            else msgspec.to_builtins(accept_all_current(self.legal_registry, today=_today()))
        )
        return await client.post(
            "/v1/auth/oauth/complete-signup",
            json={"ticket": ticket, "accepted_documents": submission},
            headers=VEX,
        )


def build_service(
    session: AsyncSession,
    *,
    store: OAuthFlowStore,
    provider: FakeProviderClient,
    legal_registry: LegalDocumentRegistry,
    jwt_service: JWTService,
    ops: Any = None,
) -> OAuthService:
    settings = oauth_settings()
    legal = LegalAcceptanceService(
        registry=legal_registry, repository=LegalAcceptanceRepository(session), session=session
    )
    auth = AuthService(
        repository=UserRepository(session),
        jwt_service=jwt_service,
        password_service=PasswordService(),
        token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
        legal_acceptance_service=legal,
        session=session,
        ops_event_bus=ops,
    )
    return OAuthService(
        registry=fake_registry(settings, provider),
        store=store,
        identity_repo=UserIdentityRepository(session),
        user_repo=UserRepository(session),
        auth_service=auth,
        session=session,
        settings=settings,
    )


async def _count(session: AsyncSession, model: Any) -> int:
    return int((await session.execute(select(func.count()).select_from(model))).scalar_one())


async def _identities(session: AsyncSession, subject: str = SUBJECT) -> list[UserIdentity]:
    result = await session.execute(select(UserIdentity).where(UserIdentity.subject == subject))
    return list(result.scalars().all())


async def _verified_user(make_user: UserFactory, session: AsyncSession, **kwargs: Any) -> User:
    user = await make_user(**kwargs)
    user.email_verified_at = datetime.now(UTC)
    await session.flush()
    return user


async def _link(session: AsyncSession, user: User, subject: str = SUBJECT) -> UserIdentity:
    row = UserIdentity(
        id=new_id(),
        user_id=user.id,
        product_id=user.product_id,
        provider=OAuthProvider.GOOGLE,
        subject=subject,
    )
    session.add(row)
    await session.flush()
    return row


@pytest_asyncio.fixture
async def redis_store() -> AsyncGenerator[RedisOAuthFlowStore]:
    if not _REDIS_URL:
        pytest.skip("REDIS_URL not set — no real Redis available")
    client = aioredis.Redis.from_url(_REDIS_URL)
    try:
        yield RedisOAuthFlowStore(lambda: client)
    finally:
        await client.aclose()


# ---------------------------------------------------------------------------
# Signup
# ---------------------------------------------------------------------------


class TestSignup:
    async def test_callback_writes_nothing_before_consent(self, db_session: AsyncSession) -> None:
        """I7 — an unknown identity creates no rows at the callback."""
        flow = Flow(db_session)
        models = (User, UserIdentity, TokenAccount, LegalAcceptance)
        before = [await _count(db_session, m) for m in models]
        async with flow.client() as client:
            frag = await flow.callback(client)
        assert frag["result"] == "signup"
        assert [await _count(db_session, m) for m in models] == before

    async def test_completed_signup_rows(self, db_session: AsyncSession) -> None:
        """I9 + I14 — rows, verified email, SIGNUP acceptances, account, lgl."""
        flow = Flow(db_session, email="new.person@example.com")
        async with flow.client() as client:
            ticket = (await flow.callback(client))["ticket"]
            info = await client.post(
                "/v1/auth/oauth/signup-info", json={"ticket": ticket}, headers=VEX
            )
            assert info.json() == {"email": "new.person@example.com", "provider": "google"}
            resp = await flow.complete(client, ticket)
        assert resp.status_code == 201, resp.text

        user = (
            await db_session.execute(select(User).where(User.email == "new.person@example.com"))
        ).scalar_one()
        assert user.password_hash is None
        assert user.email_verified_at is not None
        assert user.product_id == "vex"
        (linked,) = await _identities(db_session)
        assert linked.user_id == user.id
        assert linked.product_id == "vex"
        acceptances = (
            (
                await db_session.execute(
                    select(LegalAcceptance).where(LegalAcceptance.user_id == user.id)
                )
            )
            .scalars()
            .all()
        )
        assert len(acceptances) == 3
        assert {a.source for a in acceptances} == {LegalAcceptanceSource.SIGNUP}
        assert {a.ip_address for a in acceptances} == {CONTEXT.ip_address}
        account = (
            await db_session.execute(select(TokenAccount).where(TokenAccount.user_id == user.id))
        ).scalar_one()
        assert account.account_type == "personal"
        flow.ops.publish.assert_awaited_once()
        assert flow.ops.publish.await_args.kwargs["event_type"] == OpsEventType.USER_REGISTERED

        # I14 — the access token carries the digest the guard requires.
        access = resp.json()["access_token"]
        claims = pyjwt.decode(access, options={"verify_signature": False})
        required = flow.legal_registry.required_digest(VEX_CONFIG, today=_today())
        assert claims["lgl"] == required
        payload = flow.jwt.decode_access_token(access)
        assert payload is not None
        connection = MagicMock()
        connection.scope = {"method": "POST"}
        connection.state = {"product_config": VEX_CONFIG}
        connection.app.state = {"legal_registry": flow.legal_registry}
        handler = MagicMock()
        handler.opt = {}
        _enforce_legal_acceptance(connection, handler, payload)  # no 428

    async def test_stale_legal_keeps_ticket(self, db_session: AsyncSession) -> None:
        """I8 — 409 stale, ticket survives, retry with current docs → 201."""
        flow = Flow(db_session)
        current = accept_all_current(flow.legal_registry, today=_today())
        stale = [
            {"doc_type": d.doc_type, "version": "2019-01-01", "sha256": d.sha256} for d in current
        ]
        async with flow.client() as client:
            ticket = (await flow.callback(client))["ticket"]
            resp = await flow.complete(client, ticket, stale)
            assert resp.status_code == 409
            assert resp.json()["error"] == "legal_version_stale"
            assert not await _identities(db_session)
            info = await client.post(
                "/v1/auth/oauth/signup-info", json={"ticket": ticket}, headers=VEX
            )
            assert info.status_code == 200
            retry = await flow.complete(client, ticket)
        assert retry.status_code == 201

    async def test_ticket_single_use(self, db_session: AsyncSession) -> None:
        flow = Flow(db_session)
        async with flow.client() as client:
            ticket = (await flow.callback(client))["ticket"]
            assert (await flow.complete(client, ticket)).status_code == 201
            again = await flow.complete(client, ticket)
        assert again.status_code == 400
        assert again.json()["error"] == "invalid_signup_ticket"

    async def test_complete_signup_other_browser_rejected(self, db_session: AsyncSession) -> None:
        """I6 — a ticket replayed from another browser (other binding) is refused."""
        flow = Flow(db_session)
        async with flow.client() as client:
            ticket = (await flow.callback(client))["ticket"]
            client.cookies.clear()
            client.cookies.set(OAUTH_TX_COOKIE, "attacker-binding-value")
            resp = await flow.complete(client, ticket)
        assert resp.status_code == 400
        assert resp.json()["error"] == "invalid_signup_ticket"
        assert not await _identities(db_session)


class TestSignupIdentityConflict:
    async def test_subject_linked_since_callback_rolls_back_whole_signup(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """A unique violation on the identity insert leaves no user/account/acceptances."""
        flow = Flow(db_session, email="late@example.com")
        async with flow.client() as client:
            ticket = (await flow.callback(client))["ticket"]
            other = await _verified_user(make_user, db_session, email="other@example.com")
            await _link(db_session, other)  # same (vex, google, SUBJECT)
            models = (User, TokenAccount, LegalAcceptance)
            before = [await _count(db_session, m) for m in models]
            resp = await flow.complete(client, ticket)
        assert resp.status_code == 409
        assert resp.json()["error"] == "identity_conflict"
        assert [await _count(db_session, m) for m in models] == before
        late = await db_session.execute(select(User).where(User.email == "late@example.com"))
        assert late.scalar_one_or_none() is None


class TestConcurrentSignup:
    """I5 — concurrent complete-signup on one ticket produces exactly one account."""

    async def test_one_winner_memory_store(self, db_engine: AsyncEngine) -> None:
        await self._race(db_engine, InMemoryOAuthFlowStore())

    async def test_one_winner_redis_store(
        self, db_engine: AsyncEngine, redis_store: RedisOAuthFlowStore
    ) -> None:
        await self._race(db_engine, redis_store)

    async def _race(self, db_engine: AsyncEngine, store: OAuthFlowStore) -> None:
        from sqlalchemy.ext.asyncio import AsyncSession

        email = f"race-{uuid4().hex}@example.com"
        subject = f"race-sub-{uuid4().hex}"
        ticket = f"ticket{uuid4().hex}"
        binding = "race-binding"
        await store.put_signup(
            ticket,
            PendingSignup(
                product_id="vex",
                provider=OAuthProvider.GOOGLE,
                subject=subject,
                email=email,
                binding_hash=binding_hash(binding),
            ),
            600,
        )
        registry = make_legal_registry()
        documents = accept_all_current(registry, today=_today())
        provider = FakeProviderClient(identity(subject=subject, email=email))
        jwt_service = JWTService(JWTConfig(secret_key=JWT_SECRET))

        async def attempt() -> UUID | None:
            async with AsyncSession(bind=db_engine, expire_on_commit=False) as session:
                service = build_service(
                    session,
                    store=store,
                    provider=provider,
                    legal_registry=registry,
                    jwt_service=jwt_service,
                )
                try:
                    user, _tokens = await service.complete_signup(
                        product=VEX_CONFIG,
                        ticket=ticket,
                        binding=binding,
                        accepted_documents=documents,
                        display_name=None,
                        context=CONTEXT,
                    )
                except InvalidSignupTicketError:
                    await session.rollback()
                    return None
                await session.commit()
                return user.id

        try:
            results = await asyncio.gather(*(attempt() for _ in range(4)))
            winners = [r for r in results if r is not None]
            assert len(winners) == 1
            async with AsyncSession(bind=db_engine) as session:
                users = (
                    (await session.execute(select(User).where(User.email == email))).scalars().all()
                )
                assert [u.id for u in users] == winners
        finally:
            async with AsyncSession(bind=db_engine) as session:
                await session.execute(delete(User).where(User.email == email))
                await session.commit()


# ---------------------------------------------------------------------------
# Login / linking
# ---------------------------------------------------------------------------


class TestLogin:
    async def test_linked_identity_logs_in_and_exchange_is_single_use(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """I5 (handoff single use) + login through an existing identity."""
        user = await _verified_user(make_user, db_session, email="person@example.com")
        linked = await _link(db_session, user)
        flow = Flow(db_session)
        async with flow.client() as client:
            frag = await flow.callback(client)
            assert frag["result"] == "login"
            first = await client.post(
                "/v1/auth/oauth/exchange", json={"code": frag["code"]}, headers=VEX
            )
            second = await client.post(
                "/v1/auth/oauth/exchange", json={"code": frag["code"]}, headers=VEX
            )
        assert first.status_code == 200
        payload = flow.jwt.decode_access_token(first.json()["access_token"])
        assert payload is not None
        assert payload.user_id == user.id
        assert second.status_code == 400
        assert second.json()["error"] == "invalid_handoff"
        await db_session.refresh(linked)
        assert linked.last_login_at is not None

    async def test_exchange_from_other_browser_rejected(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """I6 — login-CSRF: a handoff code redeemed with another binding is refused."""
        user = await _verified_user(make_user, db_session, email="person@example.com")
        await _link(db_session, user)
        flow = Flow(db_session)
        async with flow.client() as client:
            code = (await flow.callback(client))["code"]
            client.cookies.clear()
            client.cookies.set(OAUTH_TX_COOKIE, "attacker-binding-value")
            with structlog.testing.capture_logs() as logs:
                resp = await client.post(
                    "/v1/auth/oauth/exchange", json={"code": code}, headers=VEX
                )
        assert resp.status_code == 400
        assert any(e["event"] == "auth.oauth.exchange_rejected" for e in logs)

    async def test_auto_link_verified_account(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """I10 — both sides verified → link + login."""
        user = await _verified_user(make_user, db_session, email="person@example.com")
        flow = Flow(db_session, email="person@example.com")
        async with flow.client() as client:
            frag = await flow.callback(client)
        assert frag["result"] == "login"
        (linked,) = await _identities(db_session)
        assert linked.user_id == user.id
        assert linked.last_login_at is not None

    async def test_unverified_local_account_not_linked(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """I10 — local email unverified → account_exists_unverified, no link."""
        await make_user(email="person@example.com")
        flow = Flow(db_session)
        async with flow.client() as client:
            frag = await flow.callback(client)
        assert frag == {"result": "error", "error": "account_exists_unverified"}
        assert not await _identities(db_session)

    async def test_account_linked_to_other_subject_conflicts(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """I10 — the local account already has a different Google subject."""
        user = await _verified_user(make_user, db_session, email="person@example.com")
        await _link(db_session, user, subject="some-other-subject")
        flow = Flow(db_session)
        async with flow.client() as client:
            frag = await flow.callback(client)
        assert frag == {"result": "error", "error": "identity_conflict"}
        assert not await _identities(db_session)

    async def test_other_product_account_is_not_linked(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """Accounts are product-scoped: a verified synthara account is invisible to vex."""
        await _verified_user(
            make_user, db_session, email="person@example.com", product_id="synthara"
        )
        flow = Flow(db_session)
        async with flow.client() as client:
            frag = await flow.callback(client)
        assert frag["result"] == "signup"

    async def test_admin_deactivated_identity_is_inactive(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """I11 — a kept identity of a deactivated user never starts a signup."""
        user = await _verified_user(make_user, db_session, email="person@example.com")
        await _link(db_session, user)
        user.is_active = False
        await db_session.flush()
        flow = Flow(db_session)
        async with flow.client() as client:
            frag = await flow.callback(client)
        assert frag == {"result": "error", "error": "account_inactive"}
        assert isinstance(flow.store, InMemoryOAuthFlowStore)
        assert flow.store.signups == {}
        assert flow.store.handoffs == {}


class TestCallbackRejections:
    """I4 — every broken flow redirects with flow_expired, and the provider is never called."""

    @pytest.mark.parametrize("case", ["no_cookie", "other_cookie", "replay", "product"])
    async def test_flow_expired(self, db_session: AsyncSession, case: str) -> None:
        flow = Flow(db_session)
        async with flow.client() as client:
            resp = await client.get(
                "/v1/auth/oauth/google/authorize", headers=VEX, follow_redirects=False
            )
            state = query_params(resp.headers["location"])["state"]
            headers = dict(VEX)
            if case == "no_cookie":
                client.cookies.clear()
            elif case == "other_cookie":
                client.cookies.clear()
                client.cookies.set(OAUTH_TX_COOKIE, "attacker-binding-value")
            elif case == "replay":
                await client.get(
                    "/v1/auth/oauth/google/callback",
                    params={"code": "c", "state": state},
                    headers=VEX,
                    follow_redirects=False,
                )
                flow.provider.exchanges.clear()
            else:
                headers = {"X-Product-Id": "synthara"}
            with structlog.testing.capture_logs() as logs:
                cb = await client.get(
                    "/v1/auth/oauth/google/callback",
                    params={"code": "c", "state": state},
                    headers=headers,
                    follow_redirects=False,
                )
        assert fragment_params(cb.headers["location"]) == {
            "result": "error",
            "error": "flow_expired",
        }
        assert flow.provider.exchanges == []
        (rejected,) = [e for e in logs if e["event"] == "auth.oauth.callback_rejected"]
        assert (
            rejected["reason"]
            == {
                "no_cookie": "missing_binding",
                "other_cookie": "binding_mismatch",
                "replay": "unknown_state",
                "product": "product_mismatch",
            }[case]
        )


# ---------------------------------------------------------------------------
# Self-closure / password reset
# ---------------------------------------------------------------------------


class TestSelfClosure:
    async def test_closure_unlinks_and_subject_can_sign_up_again(
        self, db_session: AsyncSession, make_user: UserFactory
    ) -> None:
        """I12 — identities deleted in the closure transaction; re-signup yields a new user."""
        old = await _verified_user(make_user, db_session, email="person@example.com")
        await _link(db_session, old)
        registry = make_legal_registry()
        users = UserService(
            repository=UserRepository(db_session),
            password_service=PasswordService(),
            age_verification_service=MagicMock(),
            token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
            legal_acceptance_service=LegalAcceptanceService(
                registry=registry,
                repository=LegalAcceptanceRepository(db_session),
                session=db_session,
            ),
            identity_repository=UserIdentityRepository(db_session),
            session=db_session,
        )
        await users.deactivate_account(old.id, product=VEX_CONFIG, context=CONTEXT)
        assert not await _identities(db_session)

        flow = Flow(db_session)
        async with flow.client() as client:
            frag = await flow.callback(client)
            assert frag["result"] == "signup"
            resp = await flow.complete(client, frag["ticket"])
        assert resp.status_code == 201
        (linked,) = await _identities(db_session)
        assert linked.user_id != old.id


class TestResetPasswordVerifiesEmail:
    """I17 — a consumed reset link marks the email verified (never overwrites)."""

    def _service(self) -> EmailVerificationService:
        return EmailVerificationService(
            email_service=MagicMock(),
            app_url="https://vex.test",
            token_revocation_service=TokenRevocationService(None, max_token_ttl_seconds=0),
        )

    async def test_sets_when_null(
        self, db_session: AsyncSession, make_user: UserFactory, make_reset_token: ResetTokenFactory
    ) -> None:
        user = await make_user(email=f"reset-{uuid4().hex[:8]}@example.com")
        _row, raw = await make_reset_token(user=user)
        await self._service().reset_password(raw, "new-password-123", session=db_session)
        await db_session.refresh(user)
        assert user.email_verified_at is not None

    async def test_keeps_existing(
        self, db_session: AsyncSession, make_user: UserFactory, make_reset_token: ResetTokenFactory
    ) -> None:
        user = await make_user(email=f"reset-{uuid4().hex[:8]}@example.com")
        original = datetime(2024, 1, 2, 3, 4, 5, tzinfo=UTC)
        user.email_verified_at = original
        await db_session.flush()
        _row, raw = await make_reset_token(user=user)
        await self._service().reset_password(raw, "new-password-123", session=db_session)
        await db_session.refresh(user)
        assert user.email_verified_at == original


# ---------------------------------------------------------------------------
# Migration 048
# ---------------------------------------------------------------------------

_MIGRATION_PATH = (
    Path(__file__).resolve().parents[2] / "alembic" / "versions" / "048_user_identities.py"
)


def _load_migration() -> ModuleType:
    spec = importlib.util.spec_from_file_location("revision_048", _MIGRATION_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestMigration048:
    """I23 — round-trips cleanly; downgrade refuses to fabricate password hashes."""

    async def test_round_trip(self, db_engine: AsyncEngine) -> None:
        migration = _load_migration()
        async with db_engine.connect() as connection:
            transaction = await connection.begin()
            try:

                def _round_trip(sync_connection: Connection) -> None:
                    context = MigrationContext.configure(sync_connection)
                    with Operations.context(context):
                        migration.downgrade()
                        migration.upgrade()

                await connection.run_sync(_round_trip)
                nullable = await connection.execute(
                    text(
                        "SELECT is_nullable FROM information_schema.columns "
                        "WHERE table_name = 'users' AND column_name = 'password_hash'"
                    )
                )
                assert nullable.scalar_one() == "YES"
                constraints = await connection.execute(
                    text(
                        "SELECT conname FROM pg_constraint "
                        "WHERE conrelid = 'user_identities'::regclass AND contype = 'u'"
                    )
                )
                assert set(constraints.scalars().all()) == {
                    "uq_user_identities_product_provider_subject",
                    "uq_user_identities_user_provider",
                }
            finally:
                await transaction.rollback()

    async def test_downgrade_refuses_passwordless_users(self, db_engine: AsyncEngine) -> None:
        migration = _load_migration()
        async with db_engine.connect() as connection:
            transaction = await connection.begin()
            try:
                await connection.execute(
                    text(
                        "INSERT INTO users (id, email, password_hash, product_id) "
                        "VALUES (:id, :email, NULL, 'vex')"
                    ),
                    {"id": uuid4(), "email": f"oauth-only-{uuid4().hex}@example.com"},
                )

                def _downgrade(sync_connection: Connection) -> None:
                    context = MigrationContext.configure(sync_connection)
                    with Operations.context(context):
                        migration.downgrade()

                with pytest.raises(RuntimeError, match=r"password_hash IS NULL"):
                    await connection.run_sync(_downgrade)
            finally:
                await transaction.rollback()


class TestRedisFlowStore:
    """``SET … EX`` + ``GETDEL`` semantics against a real Redis (bytes and str pools)."""

    @pytest.mark.parametrize("decode_responses", [False, True])
    async def test_single_use_with_ttl(self, decode_responses: bool) -> None:
        from src.api.services.oauth.flow_store import FLOW_PREFIX, SIGNUP_PREFIX
        from src.api.services.oauth.models import OAuthFlow

        if not _REDIS_URL:
            pytest.skip("REDIS_URL not set — no real Redis available")
        client = aioredis.Redis.from_url(_REDIS_URL, decode_responses=decode_responses)
        store = RedisOAuthFlowStore(lambda: client)
        state, ticket = f"s{uuid4().hex}", f"t{uuid4().hex}"
        flow = OAuthFlow(
            product_id="vex",
            provider=OAuthProvider.GOOGLE,
            code_verifier="v" * 64,
            nonce="n",
            binding_hash="h",
            return_to="/library",
        )
        signup = PendingSignup(
            product_id="vex",
            provider=OAuthProvider.GOOGLE,
            subject="sub",
            email="e@example.com",
            binding_hash="h",
        )
        try:
            await store.put_flow(state, flow, 600)
            await store.put_signup(ticket, signup, 900)
            assert 0 < await client.ttl(FLOW_PREFIX + state) <= 600
            assert 600 < await client.ttl(SIGNUP_PREFIX + ticket) <= 900

            assert await store.take_flow(state) == flow
            assert await store.take_flow(state) is None

            assert await store.peek_signup(ticket) == signup
            assert await store.peek_signup(ticket) == signup
            assert await store.take_signup(ticket) == signup
            assert await store.take_signup(ticket) is None
            assert await store.take_handoff("never-issued") is None
        finally:
            await client.delete(FLOW_PREFIX + state, SIGNUP_PREFIX + ticket)
            await client.aclose()
