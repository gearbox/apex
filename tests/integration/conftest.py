"""Shared fixtures for integration tests against a real PostgreSQL database.

Fixture scope hierarchy:
  session-scope : async engine + run all Alembic migrations once
  function-scope: async session wrapped in a SAVEPOINT transaction
                  that is rolled back after each test (fast isolation)
"""

from __future__ import annotations

import hashlib
import os
import secrets
import subprocess
from collections.abc import AsyncGenerator, Callable, Coroutine
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from src.db.models.auth_tokens import EmailVerificationToken, PasswordResetToken
from src.db.models.billing import Organization, TokenAccount
from src.db.models.storage import GenerationJob, UserImage
from src.db.models.user import User
from src.db.repositories.auth_tokens import AuthTokenRepository
from src.db.repositories.billing import BillingRepository
from src.db.repositories.generation_model import GenerationModelRepository
from src.db.repositories.storage import StorageRepository
from src.db.repositories.user import UserRepository

# ---------------------------------------------------------------------------
# Test database URL
# ---------------------------------------------------------------------------

_DEFAULT_TEST_DB_URL = "postgresql+asyncpg://apex_test:apex_test@localhost:5433/apex_test"
TEST_DATABASE_URL = os.environ.get("TEST_DATABASE_URL", _DEFAULT_TEST_DB_URL)

# ---------------------------------------------------------------------------
# Callable aliases for factory fixtures
# ---------------------------------------------------------------------------

UserFactory = Callable[..., Coroutine[Any, Any, User]]
OrgFactory = Callable[..., Coroutine[Any, Any, Organization]]
TokenAccountFactory = Callable[..., Coroutine[Any, Any, TokenAccount]]
JobFactory = Callable[..., Coroutine[Any, Any, GenerationJob]]
UserImageFactory = Callable[..., Coroutine[Any, Any, UserImage]]
VerificationTokenFactory = Callable[..., Coroutine[Any, Any, tuple[EmailVerificationToken, str]]]
ResetTokenFactory = Callable[..., Coroutine[Any, Any, tuple[PasswordResetToken, str]]]


# ---------------------------------------------------------------------------
# Session-scoped engine + schema bootstrap
# ---------------------------------------------------------------------------


def _run_migrations(db_url: str) -> None:
    """Run ``alembic upgrade head`` synchronously via subprocess.

    Sets DATABASE_URL env var so alembic/env.py picks it up via pydantic-settings.
    The URL must use asyncpg because alembic/env.py uses async_engine_from_config.
    """
    env = {
        **os.environ,
        "DATABASE_URL": db_url,
        # Provide minimal required settings so pydantic-settings validation passes
        "JWT_SECRET_KEY": os.environ.get(
            "JWT_SECRET_KEY", "test-secret-key-for-integration-tests-only-32b"
        ),
        "COMFYUI_HOST": os.environ.get("COMFYUI_HOST", "127.0.0.1"),
        "COMFYUI_PORT": os.environ.get("COMFYUI_PORT", "18188"),
        "R2_ACCOUNT_ID": os.environ.get("R2_ACCOUNT_ID", "test"),
        "R2_ACCESS_KEY_ID": os.environ.get("R2_ACCESS_KEY_ID", "test"),
        "R2_SECRET_ACCESS_KEY": os.environ.get("R2_SECRET_ACCESS_KEY", "test"),
        "R2_BUCKET_NAME": os.environ.get("R2_BUCKET_NAME", "test-bucket"),
    }
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Alembic upgrade failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )


@pytest_asyncio.fixture(scope="session")
async def db_engine() -> AsyncGenerator[AsyncEngine]:
    """Create async engine; run Alembic migrations once; yield; dispose."""
    _run_migrations(TEST_DATABASE_URL)

    engine = create_async_engine(TEST_DATABASE_URL, echo=False, poolclass=NullPool)
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
async def storage_repo(db_session: AsyncSession) -> StorageRepository:
    """StorageRepository bound to the test session."""
    return StorageRepository(db_session)


@pytest_asyncio.fixture
async def auth_token_repo(db_session: AsyncSession) -> AuthTokenRepository:
    """AuthTokenRepository bound to the test session."""
    return AuthTokenRepository(db_session)


@pytest_asyncio.fixture
async def generation_model_repo(db_session: AsyncSession) -> GenerationModelRepository:
    """GenerationModelRepository bound to the test session."""
    return GenerationModelRepository(db_session)


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
    ) -> User:
        user = User(
            id=user_id or uuid4(),
            email=email.lower(),
            password_hash=password_hash,
            display_name=display_name,
            is_active=is_active,
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
    ) -> Organization:
        if owner is None:
            owner = await make_user(email=f"owner-{uuid4().hex[:8]}@example.com")
        org = Organization(
            id=org_id or uuid4(),
            name=name,
            slug=slug or f"test-org-{uuid4().hex[:8]}",
            owner_id=owner.id,
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
    ) -> TokenAccount:
        if account_type == "personal":
            if user is None:
                user = await make_user(email=f"billing-{uuid4().hex[:8]}@example.com")
            account = TokenAccount(
                id=account_id or uuid4(),
                account_type="personal",
                user_id=user.id,
            )
        else:
            if org is None:
                org = await make_org()
            account = TokenAccount(
                id=account_id or uuid4(),
                account_type="enterprise",
                organization_id=org.id,
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
        provider: str = "comfyui",
        model: str | None = None,
        job_id: UUID | None = None,
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
        )
        db_session.add(job)
        await db_session.flush()
        return job

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
