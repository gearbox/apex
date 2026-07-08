"""Shared fixtures for integration tests against a real PostgreSQL database.

Fixture scope hierarchy:
  session-scope : async engine + run all Alembic migrations once
  function-scope: async session wrapped in a SAVEPOINT transaction
                  that is rolled back after each test (fast isolation)
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import subprocess
import time
from collections.abc import AsyncGenerator, Callable, Coroutine, Generator
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from src.db.models.auth_tokens import EmailVerificationToken, PasswordResetToken
from src.db.models.billing import Organization, TokenAccount
from src.db.models.gpu_session import GpuSession
from src.db.models.storage import GenerationJob, UserImage
from src.db.models.user import User
from src.db.repositories.auth_tokens import AuthTokenRepository
from src.db.repositories.billing import BillingRepository
from src.db.repositories.generation_model import GenerationModelRepository
from src.db.repositories.job import JobRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.push_subscription import PushSubscriptionRepository
from src.db.repositories.user import UserRepository
from src.db.repositories.user_image import UserImageRepository

# ---------------------------------------------------------------------------
# Test database URL
# ---------------------------------------------------------------------------

_POSTGRES_IMAGE = os.environ.get("TEST_POSTGRES_IMAGE", "postgres:16-alpine")
_POSTGRES_USER = "apex_test"
_POSTGRES_PASSWORD = "apex_test"
_POSTGRES_DB = "apex_test"
_POSTGRES_CONTAINER_PORT = "5432/tcp"
_POSTGRES_HOST = "127.0.0.1"
_POSTGRES_HEALTH_TIMEOUT_SECONDS = 60.0
_DEFAULT_TEST_DB_URL = (
    f"postgresql+asyncpg://{_POSTGRES_USER}:{_POSTGRES_PASSWORD}@localhost:5433/{_POSTGRES_DB}"
)
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", _DEFAULT_TEST_DB_URL)

# ---------------------------------------------------------------------------
# Callable aliases for factory fixtures
# ---------------------------------------------------------------------------

UserFactory = Callable[..., Coroutine[Any, Any, User]]
OrgFactory = Callable[..., Coroutine[Any, Any, Organization]]
TokenAccountFactory = Callable[..., Coroutine[Any, Any, TokenAccount]]
JobFactory = Callable[..., Coroutine[Any, Any, GenerationJob]]
GpuSessionFactory = Callable[..., Coroutine[Any, Any, GpuSession]]
UserImageFactory = Callable[..., Coroutine[Any, Any, UserImage]]
VerificationTokenFactory = Callable[..., Coroutine[Any, Any, tuple[EmailVerificationToken, str]]]
ResetTokenFactory = Callable[..., Coroutine[Any, Any, tuple[PasswordResetToken, str]]]


# ---------------------------------------------------------------------------
# Session-scoped engine + schema bootstrap
# ---------------------------------------------------------------------------


def _run_docker(args: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a docker CLI command and surface useful diagnostics on failure."""
    command = ["docker", *args]
    result = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        rendered = " ".join(command)
        raise RuntimeError(
            f"Docker command failed ({rendered}):\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result


def _container_logs(container_name: str) -> str:
    result = _run_docker(["logs", "--tail", "100", container_name], check=False)
    return f"stdout: {result.stdout}\nstderr: {result.stderr}"


def _inspect_container(container_name: str) -> dict[str, Any]:
    result = _run_docker(["inspect", container_name])
    if inspected := json.loads(result.stdout):
        return inspected[0]
    raise RuntimeError(f"Docker container {container_name!r} was not found.")


def _wait_for_postgres_health(container_name: str) -> None:
    deadline = time.monotonic() + _POSTGRES_HEALTH_TIMEOUT_SECONDS
    last_status = "unknown"

    while time.monotonic() < deadline:
        state = _inspect_container(container_name).get("State", {})
        container_status = state.get("Status", "unknown")
        health_status = state.get("Health", {}).get("Status", container_status)
        last_status = str(health_status)

        if health_status == "healthy":
            return
        if container_status == "exited":
            raise RuntimeError(
                f"PostgreSQL test container exited before becoming healthy.\n"
                f"{_container_logs(container_name)}"
            )

        time.sleep(0.5)

    raise RuntimeError(
        f"PostgreSQL test container did not become healthy within "
        f"{_POSTGRES_HEALTH_TIMEOUT_SECONDS:.0f}s; last status: {last_status}.\n"
        f"{_container_logs(container_name)}"
    )


def _host_port(container_name: str) -> str:
    ports = _inspect_container(container_name).get("NetworkSettings", {}).get("Ports", {})
    if bindings := ports.get(_POSTGRES_CONTAINER_PORT):
        return str(bindings[0]["HostPort"])
    raise RuntimeError(
        f"Docker did not publish {_POSTGRES_CONTAINER_PORT} for {container_name}."
    )


def _remove_postgres_container(container_name: str) -> None:
    _run_docker(["rm", "-f", "-v", container_name], check=False)


def _start_postgres_container() -> tuple[str, str]:
    _run_docker(["info"])
    _run_docker(["pull", _POSTGRES_IMAGE])

    container_name = f"apex-postgres-test-{os.getpid()}-{secrets.token_hex(4)}"
    try:
        _run_docker(
            [
                "run",
                "--detach",
                "--name",
                container_name,
                "--env",
                f"POSTGRES_USER={_POSTGRES_USER}",
                "--env",
                f"POSTGRES_PASSWORD={_POSTGRES_PASSWORD}",
                "--env",
                f"POSTGRES_DB={_POSTGRES_DB}",
                "--tmpfs",
                "/var/lib/postgresql/data",
                "--publish",
                f"{_POSTGRES_HOST}::5432",
                "--health-cmd",
                f"pg_isready -U {_POSTGRES_USER} -d {_POSTGRES_DB}",
                "--health-interval",
                "1s",
                "--health-timeout",
                "3s",
                "--health-retries",
                "60",
                "--health-start-period",
                "2s",
                _POSTGRES_IMAGE,
            ]
        )
        _wait_for_postgres_health(container_name)
        return container_name, _host_port(container_name)
    except Exception:
        _remove_postgres_container(container_name)
        raise


@pytest.fixture(scope="session")
def test_database_url() -> Generator[str]:
    """Provide the integration database URL, managing Docker unless one is supplied."""
    if explicit_database_url := os.environ.get("TEST_DATABASE_URL"):
        yield explicit_database_url
        return

    container_name, host_port = _start_postgres_container()
    db_url = (
        f"postgresql+asyncpg://{_POSTGRES_USER}:{_POSTGRES_PASSWORD}@"
        f"{_POSTGRES_HOST}:{host_port}/{_POSTGRES_DB}"
    )
    previous_test_database_url = os.environ.get("TEST_DATABASE_URL")
    previous_database_url = os.environ.get("DATABASE_URL")

    global TEST_DATABASE_URL  # noqa: PLW0603
    TEST_DATABASE_URL = db_url
    os.environ["TEST_DATABASE_URL"] = db_url
    os.environ["DATABASE_URL"] = db_url

    try:
        yield db_url
    finally:
        if previous_test_database_url is None:
            os.environ.pop("TEST_DATABASE_URL", None)
        else:
            os.environ["TEST_DATABASE_URL"] = previous_test_database_url
        if previous_database_url is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous_database_url
        _remove_postgres_container(container_name)


def _run_migrations(db_url: str) -> None:
    """Run ``alembic upgrade head`` synchronously via subprocess.

    Sets DATABASE_URL env var so alembic/env.py picks it up via pydantic-settings.
    The URL must use asyncpg because alembic/env.py uses async_engine_from_config.
    """
    env = {
        **os.environ,
        "DATABASE_URL": db_url,
    }
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Alembic upgrade failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest_asyncio.fixture(scope="session")
async def db_engine(test_database_url: str) -> AsyncGenerator[AsyncEngine]:
    """Create async engine; run Alembic migrations once; yield; dispose."""
    _run_migrations(test_database_url)

    engine = create_async_engine(test_database_url, echo=False, poolclass=NullPool)
    yield engine
    await engine.dispose()


# ---------------------------------------------------------------------------
# Per-test session using nested SAVEPOINT for isolation
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def db_session(db_engine: AsyncEngine) -> AsyncGenerator[AsyncSession]:
    """Per-test async session backed by a rolled-back SAVEPOINT.

    Pattern:
        conn  = await engine.connect()        # single connection
        tx    = await conn.begin()            # outer transaction
        sp    = await conn.begin_nested()     # SAVEPOINT
        session = AsyncSession(bind=conn)
        yield session
        await session.rollback()              # undo any pending work
        if sp.is_active: await sp.rollback() # roll back to SAVEPOINT
        await tx.rollback()                  # roll back outer transaction
        await conn.close()
    """
    async with db_engine.connect() as conn:
        tx = await conn.begin()
        async with AsyncSession(bind=conn, expire_on_commit=False) as session:
            nested = await conn.begin_nested()
            try:
                yield session
            finally:
                await session.rollback()
                if nested.is_active:
                    await nested.rollback()
        await tx.rollback()


# ---------------------------------------------------------------------------
# Repository fixtures (function-scoped)
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def user_repo(db_session: AsyncSession) -> UserRepository:
    """UserRepository bound to the test session."""
    return UserRepository(db_session)


@pytest_asyncio.fixture
async def billing_repo(db_session: AsyncSession) -> BillingRepository:
    """BillingRepository bound to the test session."""
    return BillingRepository(db_session)


@pytest_asyncio.fixture
async def job_repo(db_session: AsyncSession) -> JobRepository:
    """JobRepository bound to the test session."""
    return JobRepository(db_session)


@pytest_asyncio.fixture
async def output_repo(db_session: AsyncSession) -> OutputRepository:
    """OutputRepository bound to the test session."""
    return OutputRepository(db_session)


@pytest_asyncio.fixture
async def user_image_repo(db_session: AsyncSession) -> UserImageRepository:
    """UserImageRepository bound to the test session."""
    return UserImageRepository(db_session)


@pytest_asyncio.fixture
async def auth_token_repo(db_session: AsyncSession) -> AuthTokenRepository:
    """AuthTokenRepository bound to the test session."""
    return AuthTokenRepository(db_session)


@pytest_asyncio.fixture
async def generation_model_repo(db_session: AsyncSession) -> GenerationModelRepository:
    """GenerationModelRepository bound to the test session."""
    return GenerationModelRepository(db_session)


@pytest_asyncio.fixture
async def push_subscription_repo(db_session: AsyncSession) -> PushSubscriptionRepository:
    """PushSubscriptionRepository bound to the test session."""
    return PushSubscriptionRepository(db_session)


# ---------------------------------------------------------------------------
# Factory helpers — insert minimal valid rows, flush but do NOT commit
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def make_user(db_session: AsyncSession) -> UserFactory:
    """Factory fixture: create a User row and flush it."""

    async def _factory(
        *,
        email: str = "test@example.com",
        password_hash: str = "hashed_password",
        display_name: str | None = None,
        is_active: bool = True,
        user_id: UUID | None = None,
        product_id: str = "vex",
    ) -> User:
        user = User(
            id=user_id or uuid4(),
            email=email.lower(),
            password_hash=password_hash,
            display_name=display_name,
            is_active=is_active,
            product_id=product_id,
        )
        db_session.add(user)
        await db_session.flush()
        return user

    return _factory


@pytest_asyncio.fixture
async def make_org(db_session: AsyncSession, make_user: UserFactory) -> OrgFactory:
    """Factory fixture: create an Organization row and flush it."""

    async def _factory(
        *,
        name: str = "Test Org",
        slug: str | None = None,
        owner: User | None = None,
        org_id: UUID | None = None,
        product_id: str = "vex",
    ) -> Organization:
        if owner is None:
            owner = await make_user(email=f"owner-{uuid4().hex[:8]}@example.com")
        org = Organization(
            id=org_id or uuid4(),
            name=name,
            slug=slug or f"test-org-{uuid4().hex[:8]}",
            owner_id=owner.id,
            product_id=product_id,
        )
        db_session.add(org)
        await db_session.flush()
        return org

    return _factory


@pytest_asyncio.fixture
async def make_token_account(
    db_session: AsyncSession,
    make_user: UserFactory,
    make_org: OrgFactory,
) -> TokenAccountFactory:
    """Factory fixture: create a TokenAccount (personal or enterprise) and flush it."""

    async def _factory(
        *,
        account_type: str = "personal",
        user: User | None = None,
        org: Organization | None = None,
        account_id: UUID | None = None,
        product_id: str = "vex",
    ) -> TokenAccount:
        if account_type == "personal":
            if user is None:
                user = await make_user(email=f"billing-{uuid4().hex[:8]}@example.com")
            account = TokenAccount(
                id=account_id or uuid4(),
                account_type="personal",
                user_id=user.id,
                product_id=product_id,
            )
        else:
            if org is None:
                org = await make_org()
            account = TokenAccount(
                id=account_id or uuid4(),
                account_type="enterprise",
                organization_id=org.id,
                product_id=product_id,
            )
        db_session.add(account)
        await db_session.flush()
        return account

    return _factory


@pytest_asyncio.fixture
async def make_job(db_session: AsyncSession, make_user: UserFactory) -> JobFactory:
    """Factory fixture: create a GenerationJob row and flush it."""

    async def _factory(
        *,
        user: User | None = None,
        status: str = "pending",
        generation_type: str = "t2i",
        prompt: str = "a test image",
        name: str = "Test Job",
        provider: str = "grok",
        model: str | None = None,
        job_id: UUID | None = None,
        product_id: str = "vex",
        external_request_id: str | None = None,
        gpu_session_id: UUID | None = None,
        is_deleted: bool = False,
    ) -> GenerationJob:
        if user is None:
            user = await make_user(email=f"jobuser-{uuid4().hex[:8]}@example.com")
        job = GenerationJob(
            id=job_id or uuid4(),
            user_id=user.id,
            name=name,
            prompt=prompt,
            status=status,
            generation_type=generation_type,
            provider=provider,
            model=model,
            product_id=product_id,
            external_request_id=external_request_id,
            gpu_session_id=gpu_session_id,
            is_deleted=is_deleted,
        )
        db_session.add(job)
        await db_session.flush()
        return job

    return _factory


@pytest_asyncio.fixture
async def make_gpu_session(db_session: AsyncSession, make_user: UserFactory) -> GpuSessionFactory:
    """Factory fixture: create a GpuSession row and flush it."""

    async def _factory(
        *,
        user: User | None = None,
        status: str = "active",
        bundle_name: str = "wan_2.2_i2v",
        model_type: str = "aisha-image",
        tunnel_hostname: str | None = None,
        session_id: UUID | None = None,
        product_id: str = "vex",
    ) -> GpuSession:
        if user is None:
            user = await make_user(email=f"gpuuser-{uuid4().hex[:8]}@example.com")
        session = GpuSession(
            id=session_id or uuid4(),
            user_id=user.id,
            product_id=product_id,
            bundle_name=bundle_name,
            model_type=model_type,
            status=status,
            tunnel_hostname=tunnel_hostname,
        )
        db_session.add(session)
        await db_session.flush()
        return session

    return _factory


@pytest_asyncio.fixture
async def make_user_image(db_session: AsyncSession, make_user: UserFactory) -> UserImageFactory:
    """Factory fixture: create a UserImage row and flush it."""

    async def _factory(
        *,
        user: User | None = None,
        storage_key: str | None = None,
        original_filename: str = "test.png",
        content_type: str = "image/png",
        size_bytes: int = 1024,
        format: str = "png",
        expires_at: datetime | None = None,
        image_id: UUID | None = None,
        product_id: str = "vex",
    ) -> UserImage:
        if user is None:
            user = await make_user(email=f"imguser-{uuid4().hex[:8]}@example.com")
        img_id = image_id or uuid4()
        image = UserImage(
            id=img_id,
            user_id=user.id,
            storage_key=storage_key or f"users/{user.id}/uploads/{img_id}.png",
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
            format=format,
            expires_at=expires_at or datetime.now(UTC) + timedelta(days=7),
            product_id=product_id,
        )
        db_session.add(image)
        await db_session.flush()
        return image

    return _factory


@pytest_asyncio.fixture
async def make_verification_token(
    db_session: AsyncSession, make_user: UserFactory
) -> VerificationTokenFactory:
    """Factory fixture: insert an EmailVerificationToken row directly and flush it."""

    async def _factory(
        *,
        user: User | None = None,
        token_hash: str | None = None,
        expires_at: datetime | None = None,
        used_at: datetime | None = None,
    ) -> tuple[EmailVerificationToken, str]:
        """Returns (token_row, raw_token)."""
        if user is None:
            user = await make_user(email=f"vtoken-{uuid4().hex[:8]}@example.com")
        raw = secrets.token_urlsafe(32)
        h = token_hash or hashlib.sha256(raw.encode()).hexdigest()
        token = EmailVerificationToken(
            id=uuid4(),
            user_id=user.id,
            token_hash=h,
            expires_at=expires_at or datetime.now(UTC) + timedelta(hours=24),
            used_at=used_at,
        )
        db_session.add(token)
        await db_session.flush()
        return token, raw

    return _factory


@pytest_asyncio.fixture
async def make_reset_token(db_session: AsyncSession, make_user: UserFactory) -> ResetTokenFactory:
    """Factory fixture: insert a PasswordResetToken row directly and flush it."""

    async def _factory(
        *,
        user: User | None = None,
        token_hash: str | None = None,
        expires_at: datetime | None = None,
        used_at: datetime | None = None,
        ip_address: str | None = None,
    ) -> tuple[PasswordResetToken, str]:
        """Returns (token_row, raw_token)."""
        if user is None:
            user = await make_user(email=f"rtoken-{uuid4().hex[:8]}@example.com")
        raw = secrets.token_urlsafe(32)
        h = token_hash or hashlib.sha256(raw.encode()).hexdigest()
        token = PasswordResetToken(
            id=uuid4(),
            user_id=user.id,
            token_hash=h,
            expires_at=expires_at or datetime.now(UTC) + timedelta(minutes=30),
            used_at=used_at,
            ip_address=ip_address,
        )
        db_session.add(token)
        await db_session.flush()
        return token, raw

    return _factory
