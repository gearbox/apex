"""Integration tests for AuthTokenRepository against a real PostgreSQL database."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.db.models.auth_tokens import EmailVerificationToken, PasswordResetToken

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.repositories.auth_tokens import AuthTokenRepository

# ---------------------------------------------------------------------------
# create_verification_token
# ---------------------------------------------------------------------------


async def test_create_verification_token_returns_raw_token(
    auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """create_verification_token returns a non-empty URL-safe string."""
    user = await make_user(email=f"vtoken-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_verification_token(user.id)
    assert isinstance(raw, str)
    assert len(raw) > 10


async def test_create_verification_token_hash_stored_correctly(
    auth_token_repo: AuthTokenRepository,
    make_user,
    db_session: AsyncSession,
) -> None:
    """The stored token_hash equals SHA-256 of the raw token."""
    user = await make_user(email=f"vhash-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_verification_token(user.id)

    expected_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(EmailVerificationToken).where(EmailVerificationToken.token_hash == expected_hash)
    )
    token = result.scalar_one_or_none()
    assert token is not None
    assert token.user_id == user.id


async def test_create_verification_token_expires_in_24h(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """Verification token expires approximately 24 hours from now."""
    user = await make_user(email=f"vexpiry-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_verification_token(user.id)

    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(EmailVerificationToken).where(EmailVerificationToken.token_hash == token_hash)
    )
    token = result.scalar_one()
    now = datetime.now(UTC)
    delta = token.expires_at - now
    # Should be close to 24 hours — allow ±1 minute
    assert timedelta(hours=23, minutes=59) <= delta <= timedelta(hours=24, minutes=1)


async def test_create_verification_token_invalidates_previous(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """Creating a second token for the same user marks the first token as used."""
    user = await make_user(email=f"vinval-{uuid4().hex[:6]}@example.com")
    first_raw = await auth_token_repo.create_verification_token(user.id)
    first_hash = hashlib.sha256(first_raw.encode()).hexdigest()

    # Create a second token — should invalidate the first
    await auth_token_repo.create_verification_token(user.id)

    result = await db_session.execute(
        select(EmailVerificationToken).where(EmailVerificationToken.token_hash == first_hash)
    )
    first_token = result.scalar_one()
    assert first_token.used_at is not None


async def test_create_verification_token_second_is_consumable(
    auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """After invalidation, the newly issued token can still be consumed."""
    user = await make_user(email=f"vinval2-{uuid4().hex[:6]}@example.com")
    await auth_token_repo.create_verification_token(user.id)
    second_raw = await auth_token_repo.create_verification_token(user.id)

    user_id = await auth_token_repo.consume_verification_token(second_raw)
    assert user_id == user.id


# ---------------------------------------------------------------------------
# consume_verification_token
# ---------------------------------------------------------------------------


async def test_consume_verification_token_happy_path(
    auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """consume_verification_token returns user_id and marks token as used."""
    user = await make_user(email=f"vconsume-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_verification_token(user.id)
    user_id = await auth_token_repo.consume_verification_token(raw)
    assert user_id == user.id


async def test_consume_verification_token_sets_used_at(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """consume_verification_token sets the used_at timestamp on the token."""
    user = await make_user(email=f"vusedset-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_verification_token(user.id)
    await auth_token_repo.consume_verification_token(raw)

    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(EmailVerificationToken).where(EmailVerificationToken.token_hash == token_hash)
    )
    token = result.scalar_one()
    assert token.used_at is not None
    assert token.used_at.tzinfo is not None


async def test_consume_verification_token_unknown_returns_none(
    auth_token_repo: AuthTokenRepository,
) -> None:
    """consume_verification_token returns None for an unknown token."""
    assert await auth_token_repo.consume_verification_token("completely_random_token") is None


async def test_consume_verification_token_expired_returns_none(
    auth_token_repo: AuthTokenRepository, make_user, make_verification_token
) -> None:
    """consume_verification_token returns None for an expired token."""
    user = await make_user(email=f"vexpired-{uuid4().hex[:6]}@example.com")
    past = datetime.now(UTC) - timedelta(seconds=1)
    _, raw = await make_verification_token(user=user, expires_at=past)
    assert await auth_token_repo.consume_verification_token(raw) is None


async def test_consume_verification_token_already_used_returns_none(
    auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """consume_verification_token returns None on a second call (single-use)."""
    user = await make_user(email=f"voneuse-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_verification_token(user.id)
    first_result = await auth_token_repo.consume_verification_token(raw)
    assert first_result == user.id

    second_result = await auth_token_repo.consume_verification_token(raw)
    assert second_result is None


# ---------------------------------------------------------------------------
# create_reset_token
# ---------------------------------------------------------------------------


async def test_create_reset_token_returns_raw_token(
    auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """create_reset_token returns a non-empty URL-safe string."""
    user = await make_user(email=f"rtoken-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id)
    assert isinstance(raw, str)
    assert len(raw) > 10


async def test_create_reset_token_stores_ip_address(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """create_reset_token persists the ip_address field."""
    user = await make_user(email=f"rip-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id, ip_address="1.2.3.4")

    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    )
    token = result.scalar_one()
    assert token.ip_address == "1.2.3.4"


async def test_create_reset_token_accepts_none_ip(
    auth_token_repo: AuthTokenRepository,
    make_user,
    db_session: AsyncSession,  # noqa: ARG001
) -> None:
    """create_reset_token accepts ip_address=None."""
    user = await make_user(email=f"rnoip-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id, ip_address=None)
    assert raw is not None


async def test_create_reset_token_expires_in_30_minutes(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """Reset token expires approximately 30 minutes from now (much less than 24 h)."""
    user = await make_user(email=f"rexp-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id)

    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    )
    token = result.scalar_one()
    now = datetime.now(UTC)
    delta = token.expires_at - now
    assert timedelta(minutes=29) <= delta <= timedelta(minutes=31)


async def test_create_reset_token_expires_shorter_than_verification(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """Reset token expiry (30 min) is much shorter than verification token (24 h)."""
    user = await make_user(email=f"rshort-{uuid4().hex[:6]}@example.com")
    v_raw = await auth_token_repo.create_verification_token(user.id)
    r_raw = await auth_token_repo.create_reset_token(user.id)

    v_hash = hashlib.sha256(v_raw.encode()).hexdigest()
    r_hash = hashlib.sha256(r_raw.encode()).hexdigest()

    v_result = await db_session.execute(
        select(EmailVerificationToken).where(EmailVerificationToken.token_hash == v_hash)
    )
    v_token = v_result.scalar_one()

    r_result = await db_session.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == r_hash)
    )
    r_token = r_result.scalar_one()

    assert r_token.expires_at < v_token.expires_at


async def test_create_reset_token_invalidates_previous(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """Creating a second reset token invalidates (marks used) the first one."""
    user = await make_user(email=f"rinval-{uuid4().hex[:6]}@example.com")
    first_raw = await auth_token_repo.create_reset_token(user.id)
    first_hash = hashlib.sha256(first_raw.encode()).hexdigest()

    await auth_token_repo.create_reset_token(user.id)

    result = await db_session.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == first_hash)
    )
    first_token = result.scalar_one()
    assert first_token.used_at is not None


# ---------------------------------------------------------------------------
# consume_reset_token
# ---------------------------------------------------------------------------


async def test_consume_reset_token_happy_path(
    auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """consume_reset_token returns user_id on valid token."""
    user = await make_user(email=f"rconsume-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id)
    user_id = await auth_token_repo.consume_reset_token(raw)
    assert user_id == user.id


async def test_consume_reset_token_marks_used(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """consume_reset_token sets used_at on the token row."""
    user = await make_user(email=f"rusedset-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id)
    await auth_token_repo.consume_reset_token(raw)

    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    )
    token = result.scalar_one()
    assert token.used_at is not None


async def test_consume_reset_token_unknown_returns_none(
    auth_token_repo: AuthTokenRepository,
) -> None:
    """consume_reset_token returns None for an unknown token."""
    assert await auth_token_repo.consume_reset_token("no_such_token") is None


async def test_consume_reset_token_expired_returns_none(
    auth_token_repo: AuthTokenRepository, make_user, make_reset_token
) -> None:
    """consume_reset_token returns None for an expired token."""
    user = await make_user(email=f"rexpired-{uuid4().hex[:6]}@example.com")
    past = datetime.now(UTC) - timedelta(seconds=1)
    _, raw = await make_reset_token(user=user, expires_at=past)
    assert await auth_token_repo.consume_reset_token(raw) is None


async def test_consume_reset_token_already_used_returns_none(
    auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """consume_reset_token returns None on second call (single-use guarantee)."""
    user = await make_user(email=f"roneuse-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id)
    first = await auth_token_repo.consume_reset_token(raw)
    assert first == user.id

    second = await auth_token_repo.consume_reset_token(raw)
    assert second is None


# ---------------------------------------------------------------------------
# Constraints and invariants
# ---------------------------------------------------------------------------


async def test_duplicate_verification_token_hash_raises(
    db_session: AsyncSession, make_user
) -> None:
    """Inserting two EmailVerificationToken rows with the same hash raises IntegrityError."""
    user = await make_user(email=f"dupvhash-{uuid4().hex[:6]}@example.com")
    same_hash = "a" * 64  # 64-char hex string

    db_session.add(
        EmailVerificationToken(
            id=uuid4(),
            user_id=user.id,
            token_hash=same_hash,
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )
    )
    await db_session.flush()

    db_session.add(
        EmailVerificationToken(
            id=uuid4(),
            user_id=user.id,
            token_hash=same_hash,
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )
    )
    with pytest.raises(IntegrityError):
        await db_session.flush()


async def test_verification_token_timezone_aware(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """expires_at and (once consumed) used_at are timezone-aware datetimes."""
    user = await make_user(email=f"vtimezone-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_verification_token(user.id)
    await auth_token_repo.consume_verification_token(raw)

    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(EmailVerificationToken).where(EmailVerificationToken.token_hash == token_hash)
    )
    token = result.scalar_one()
    assert token.expires_at.tzinfo is not None
    assert token.used_at is not None
    assert token.used_at.tzinfo is not None


async def test_reset_token_timezone_aware(
    auth_token_repo: AuthTokenRepository, make_user, db_session: AsyncSession
) -> None:
    """Reset token expires_at and used_at are timezone-aware datetimes."""
    user = await make_user(email=f"rtimezone-{uuid4().hex[:6]}@example.com")
    raw = await auth_token_repo.create_reset_token(user.id)
    await auth_token_repo.consume_reset_token(raw)

    token_hash = hashlib.sha256(raw.encode()).hexdigest()
    result = await db_session.execute(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == token_hash)
    )
    token = result.scalar_one()
    assert token.expires_at.tzinfo is not None
    assert token.used_at is not None
    assert token.used_at.tzinfo is not None


async def test_cascade_delete_user_removes_verification_tokens(
    db_session: AsyncSession, auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """Deleting a User cascades to EmailVerificationToken rows."""
    user = await make_user(email=f"vcascade-{uuid4().hex[:6]}@example.com")
    await auth_token_repo.create_verification_token(user.id)

    await db_session.delete(user)
    await db_session.flush()

    result = await db_session.execute(
        select(EmailVerificationToken).where(EmailVerificationToken.user_id == user.id)
    )
    assert result.scalars().all() == []


async def test_cascade_delete_user_removes_reset_tokens(
    db_session: AsyncSession, auth_token_repo: AuthTokenRepository, make_user
) -> None:
    """Deleting a User cascades to PasswordResetToken rows."""
    user = await make_user(email=f"rcascade-{uuid4().hex[:6]}@example.com")
    await auth_token_repo.create_reset_token(user.id)

    await db_session.delete(user)
    await db_session.flush()

    result = await db_session.execute(
        select(PasswordResetToken).where(PasswordResetToken.user_id == user.id)
    )
    assert result.scalars().all() == []
