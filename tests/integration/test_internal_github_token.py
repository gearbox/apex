"""POST /v1/internal/gpu-sessions/{session_id}/github-token — a node pulls its GitHub credential.

Real PostgreSQL rows and the real controller, route table and DI wiring; the provider is
either the real ``GitHubAppTokenProvider`` (talking to a ``MockTransport`` standing in for
api.github.com), the real static provider, or a stub that fails.

Rows are committed and the app opens its own session per request: ``AsyncTestClient`` runs
handlers on a different event loop than the test, so the savepoint-wrapped ``db_session``
(whose asyncpg connection belongs to the test's loop) cannot be shared with it.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock
from uuid import UUID, uuid4

import httpx
import pytest
import pytest_asyncio
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from litestar import Litestar
from litestar.di import Provide
from litestar.testing import AsyncTestClient
from sqlalchemy import delete, event
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from structlog.testing import capture_logs

from src.api.routes.internal_gpu_session import InternalGpuSessionController
from src.api.services.github.token_provider import (
    GitHubAppTokenProvider,
    GitHubTokenProvider,
    GitHubTokenUnavailableError,
    StaticGitHubTokenProvider,
)
from src.api.services.gpu_session.github_token_service import (
    GITHUB_TOKEN_ELIGIBLE_STATUSES,
    GpuSessionGitHubTokenService,
)
from src.core.enums import GitHubAuthMode, GpuSessionStatus
from src.db.models.gpu_session import GpuSession
from src.db.models.user import User

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Awaitable, Callable, Iterator

    from sqlalchemy.ext.asyncio import AsyncEngine

    MakeSession = Callable[[GpuSessionStatus], Awaitable[UUID]]

_CALLBACK_TOKEN = "node-callback-token-for-github-token-tests"
_INSTALLATION_TOKEN = "ghs_AppInstallationTokenValue0123456789abcd"
_STATIC_PAT = "ghp_StaticPersonalAccessToken0123456789abcd"
_INELIGIBLE = [
    GpuSessionStatus.paused,
    GpuSessionStatus.stopping,
    GpuSessionStatus.stopped,
    GpuSessionStatus.failed,
]


@pytest.fixture(scope="module")
def private_pem() -> bytes:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


@pytest.fixture
def session_factory(db_engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=db_engine, expire_on_commit=False)


@pytest_asyncio.fixture
async def make_session(
    session_factory: async_sessionmaker[AsyncSession],
) -> AsyncGenerator[MakeSession]:
    """Commit a user + GPU session in the given status; delete the users on teardown."""
    user_ids: list[UUID] = []

    async def _make(status: GpuSessionStatus) -> UUID:
        user = User(
            id=uuid4(),
            email=f"github-token-{uuid4().hex}@example.com",
            password_hash="hash",
            product_id="vex",
        )
        gpu_session = GpuSession(
            id=uuid4(),
            user_id=user.id,
            product_id="vex",
            status=status,
            bundle_name="wan_2.2_i2v",
            model_type="aisha-image",
            callback_token_hash=hashlib.sha256(_CALLBACK_TOKEN.encode()).hexdigest(),
        )
        async with session_factory() as db, db.begin():
            db.add(user)
            await db.flush()
            db.add(gpu_session)
        user_ids.append(user.id)
        return gpu_session.id

    yield _make

    async with session_factory() as db:
        await db.execute(delete(User).where(User.id.in_(user_ids)))
        await db.commit()


def _app_provider(private_pem: bytes, *, expires_at: datetime) -> GitHubAppTokenProvider:
    def github(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            201,
            json={
                "token": _INSTALLATION_TOKEN,
                "expires_at": expires_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            },
        )

    return GitHubAppTokenProvider(
        app_id=12345,
        installation_id=678,
        private_key_pem=private_pem,
        repositories=("aisha", "ai-bundles"),
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(github)),
        refresh_margin=timedelta(seconds=600),
    )


class _FailingProvider:
    mode = GitHubAuthMode.app

    def __init__(self) -> None:
        self.get_token = AsyncMock(
            side_effect=GitHubTokenUnavailableError("github app token mint failed: status 502")
        )


def _client(
    session_factory: async_sessionmaker[AsyncSession], provider: GitHubTokenProvider | None
) -> AsyncTestClient[Any]:
    service = GpuSessionGitHubTokenService(provider=provider)

    async def provide_session() -> AsyncGenerator[AsyncSession]:
        async with session_factory() as db:
            yield db

    # The controller's other handlers need their own services at app construction; none
    # of them is exercised here. Each key needs its own provider callable: Litestar
    # introspects the signature (so not `Provide(MagicMock)`) and refuses one callable
    # registered under two keys.
    def unused_service() -> Provide:
        def stub() -> MagicMock:
            return MagicMock()

        return Provide(stub, sync_to_thread=False)

    app = Litestar(
        route_handlers=[InternalGpuSessionController],
        dependencies={
            "gpu_session_github_token_service": Provide(lambda: service, sync_to_thread=False),
            "session": Provide(provide_session),
            "operation_event_service": unused_service(),
            "gpu_session_command_service": unused_service(),
            "event_bus": unused_service(),
        },
    )
    return AsyncTestClient(app=app)


def _url(session_id: UUID) -> str:
    return f"/v1/internal/gpu-sessions/{session_id}/github-token"


def _bearer(token: str = _CALLBACK_TOKEN) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


@contextmanager
def _recorded_sql(engine: AsyncEngine) -> Iterator[list[str]]:
    statements: list[str] = []

    def record(_conn: Any, _cursor: Any, statement: str, *_rest: Any) -> None:
        statements.append(statement)

    event.listen(engine.sync_engine, "before_cursor_execute", record)
    try:
        yield statements
    finally:
        event.remove(engine.sync_engine, "before_cursor_execute", record)


class TestIssued:
    """A10 — valid callback token + eligible status."""

    async def test_app_mode_returns_the_installation_token_and_its_expiry(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        make_session: MakeSession,
        private_pem: bytes,
    ) -> None:
        expires_at = datetime.now(UTC).replace(microsecond=0) + timedelta(hours=1)
        session_id = await make_session(GpuSessionStatus.active)
        provider = _app_provider(private_pem, expires_at=expires_at)

        async with _client(session_factory, provider) as client:
            response = await client.post(_url(session_id), headers=_bearer())

        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        body = response.json()
        assert body["token"] == _INSTALLATION_TOKEN
        assert datetime.fromisoformat(body["expires_at"]) == expires_at

    async def test_pat_mode_returns_the_pat_with_null_expiry(
        self, session_factory: async_sessionmaker[AsyncSession], make_session: MakeSession
    ) -> None:
        session_id = await make_session(GpuSessionStatus.provisioning)

        async with _client(session_factory, StaticGitHubTokenProvider(_STATIC_PAT)) as client:
            response = await client.post(_url(session_id), headers=_bearer())

        assert response.status_code == 200, response.text
        assert response.headers["cache-control"] == "no-store"
        assert response.json() == {"token": _STATIC_PAT, "expires_at": None}

    @pytest.mark.parametrize("status", sorted(GITHUB_TOKEN_ELIGIBLE_STATUSES))
    async def test_every_eligible_status_is_served(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        make_session: MakeSession,
        status: GpuSessionStatus,
    ) -> None:
        session_id = await make_session(status)

        async with _client(session_factory, StaticGitHubTokenProvider(_STATIC_PAT)) as client:
            response = await client.post(_url(session_id), headers=_bearer())

        assert response.status_code == 200, status

    def test_the_eligible_statuses_are_exactly_the_pre_teardown_ones(self) -> None:
        assert {
            GpuSessionStatus.pending,
            GpuSessionStatus.provisioning,
            GpuSessionStatus.active,
            GpuSessionStatus.stale,
            GpuSessionStatus.resuming,
        } == GITHUB_TOKEN_ELIGIBLE_STATUSES
        assert set(GpuSessionStatus) - GITHUB_TOKEN_ELIGIBLE_STATUSES == set(_INELIGIBLE)

    async def test_the_endpoint_reads_only_and_takes_no_row_lock(
        self,
        db_engine: AsyncEngine,
        session_factory: async_sessionmaker[AsyncSession],
        make_session: MakeSession,
    ) -> None:
        session_id = await make_session(GpuSessionStatus.active)

        async with _client(session_factory, StaticGitHubTokenProvider(_STATIC_PAT)) as client:
            with _recorded_sql(db_engine) as statements:
                response = await client.post(_url(session_id), headers=_bearer())

        assert response.status_code == 200
        assert statements, "expected at least the session lookup"
        for statement in statements:
            assert statement.lstrip().upper().startswith("SELECT"), statement
            assert "FOR UPDATE" not in statement.upper(), statement


class TestUnauthorized:
    """A11 — never reveals whether a session exists."""

    async def test_unknown_session_wrong_token_and_missing_bearer_are_all_401(
        self, session_factory: async_sessionmaker[AsyncSession], make_session: MakeSession
    ) -> None:
        session_id = await make_session(GpuSessionStatus.active)

        async with _client(session_factory, StaticGitHubTokenProvider(_STATIC_PAT)) as client:
            unknown = await client.post(_url(uuid4()), headers=_bearer())
            wrong_token = await client.post(_url(session_id), headers=_bearer("not-the-token"))
            no_header = await client.post(_url(session_id))
            empty_bearer = await client.post(_url(session_id), headers={"Authorization": "Bearer "})
            other_scheme = await client.post(
                _url(session_id), headers={"Authorization": f"Basic {_CALLBACK_TOKEN}"}
            )

        for response in (unknown, wrong_token, no_header, empty_bearer, other_scheme):
            assert response.status_code == 401, response.text
            assert response.json()["error"] == "unauthorized"
            assert _STATIC_PAT not in response.text
        # Same body for "no such session" and "wrong token": existence is not revealed.
        assert unknown.json() == wrong_token.json()

    async def test_auth_is_checked_before_eligibility_and_availability(
        self, session_factory: async_sessionmaker[AsyncSession], make_session: MakeSession
    ) -> None:
        stopped = await make_session(GpuSessionStatus.stopped)
        provider = _FailingProvider()

        async with _client(session_factory, provider) as client:  # type: ignore[arg-type]
            response = await client.post(_url(stopped), headers=_bearer("not-the-token"))

        assert response.status_code == 401
        provider.get_token.assert_not_awaited()


class TestNotEligible:
    """A12."""

    @pytest.mark.parametrize("status", _INELIGIBLE)
    async def test_teardown_and_paused_states_are_409(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        make_session: MakeSession,
        status: GpuSessionStatus,
    ) -> None:
        session_id = await make_session(status)
        provider = _FailingProvider()

        async with _client(session_factory, provider) as client:  # type: ignore[arg-type]
            response = await client.post(_url(session_id), headers=_bearer())

        assert response.status_code == 409
        assert response.json()["error"] == "session_not_eligible"
        # An ineligible session never costs a GitHub call.
        provider.get_token.assert_not_awaited()


class TestUnavailable:
    """A13."""

    async def test_provider_failure_is_503_with_retry_after(
        self, session_factory: async_sessionmaker[AsyncSession], make_session: MakeSession
    ) -> None:
        session_id = await make_session(GpuSessionStatus.active)

        async with _client(session_factory, _FailingProvider()) as client:  # type: ignore[arg-type]
            response = await client.post(_url(session_id), headers=_bearer())

        assert response.status_code == 503
        assert response.headers["retry-after"] == "30"
        assert response.json()["error"] == "github_token_unavailable"
        assert "502" not in response.text  # upstream detail stays in the logs

    async def test_no_provider_is_503_with_retry_after(
        self, session_factory: async_sessionmaker[AsyncSession], make_session: MakeSession
    ) -> None:
        """Auth mode `none`: no GitHub credential is configured in this process."""
        session_id = await make_session(GpuSessionStatus.active)

        async with _client(session_factory, None) as client:
            response = await client.post(_url(session_id), headers=_bearer())

        assert response.status_code == 503
        assert response.headers["retry-after"] == "30"
        assert response.json()["error"] == "github_token_unavailable"


class TestLogsNeverContainSecrets:
    """A14."""

    async def test_no_log_record_contains_the_token_or_the_callback_token(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        make_session: MakeSession,
        private_pem: bytes,
    ) -> None:
        session_id = await make_session(GpuSessionStatus.active)
        provider = _app_provider(private_pem, expires_at=datetime.now(UTC) + timedelta(hours=1))

        with capture_logs() as logs:
            async with _client(session_factory, provider) as client:
                ok = await client.post(_url(session_id), headers=_bearer())
                rejected = await client.post(_url(session_id), headers=_bearer("wrong-token-value"))

        assert ok.status_code == 200
        assert rejected.status_code == 401
        events = {entry["event"] for entry in logs}
        # Not vacuous: the success and rejection paths and the mint itself all logged.
        assert {
            "gpu_session.github_token.issued",
            "gpu_session.github_token.rejected",
            "github.app_token.minted",
        } <= events
        rendered = json.dumps(logs, default=str)
        assert _INSTALLATION_TOKEN not in rendered
        assert _CALLBACK_TOKEN not in rendered
        assert "wrong-token-value" not in rendered
        assert "BEGIN" not in rendered  # no key material either

    async def test_failure_logs_carry_no_response_body_or_token(
        self, session_factory: async_sessionmaker[AsyncSession], make_session: MakeSession
    ) -> None:
        session_id = await make_session(GpuSessionStatus.active)

        with capture_logs() as logs:
            async with _client(session_factory, _FailingProvider()) as client:  # type: ignore[arg-type]
                await client.post(_url(session_id), headers=_bearer())

        unavailable = [e for e in logs if e["event"] == "gpu_session.github_token.unavailable"]
        assert unavailable
        assert _CALLBACK_TOKEN not in json.dumps(logs, default=str)
