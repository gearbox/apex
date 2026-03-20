"""Integration tests for UserRepository against a real PostgreSQL database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import SubscriptionTier, UserRole
from src.db.models.user import RefreshToken, User
from src.db.repositories.user import UserRepository

# ---------------------------------------------------------------------------
# create_user
# ---------------------------------------------------------------------------


async def test_create_user_returns_user(user_repo: UserRepository) -> None:
    """create_user persists and returns a User with lowercased email."""
    user = await user_repo.create_user(
        id=uuid4(),
        email="Alice@Example.Com",
        password_hash="hashed",
        product_id="vex",
        display_name="Alice",
    )
    assert user.email == "alice@example.com"
    assert user.display_name == "Alice"
    assert user.is_active is True


async def test_create_user_lowercases_email(user_repo: UserRepository) -> None:
    """create_user stores email in lower case regardless of input case."""
    uid = uuid4()
    user = await user_repo.create_user(
        id=uid,
        email="UPPER@EXAMPLE.COM",
        password_hash="x",
        product_id="vex",
    )
    assert user.email == "upper@example.com"


async def test_create_user_duplicate_email_raises(
    user_repo: UserRepository,
    make_user,
) -> None:
    """Creating two active users with the same email raises IntegrityError."""
    await make_user(email="dup@example.com")
    with pytest.raises(IntegrityError):
        await user_repo.create_user(
            id=uuid4(),
            email="dup@example.com",
            password_hash="x",
            product_id="vex",
        )


# ---------------------------------------------------------------------------
# get_user
# ---------------------------------------------------------------------------


async def test_get_user_returns_existing(user_repo: UserRepository, make_user) -> None:
    """get_user returns the user by primary key."""
    user = await make_user(email="get@example.com")
    found = await user_repo.get_user(user.id)
    assert found is not None
    assert found.id == user.id


async def test_get_user_returns_none_for_unknown_id(user_repo: UserRepository) -> None:
    """get_user returns None for an unknown UUID."""
    result = await user_repo.get_user(uuid4())
    assert result is None


# ---------------------------------------------------------------------------
# get_user_by_email
# ---------------------------------------------------------------------------


async def test_get_user_by_email_found(user_repo: UserRepository, make_user) -> None:
    """get_user_by_email performs case-insensitive lookup."""
    await make_user(email="byemail@example.com")
    found = await user_repo.get_user_by_email("byemail@example.com")
    assert found is not None
    assert found.email == "byemail@example.com"


async def test_get_user_by_email_case_insensitive(user_repo: UserRepository, make_user) -> None:
    """get_user_by_email finds user regardless of input case."""
    await make_user(email="case@example.com")
    found = await user_repo.get_user_by_email("CASE@EXAMPLE.COM")
    assert found is not None


async def test_get_user_by_email_not_found(user_repo: UserRepository) -> None:
    """get_user_by_email returns None when email does not exist."""
    result = await user_repo.get_user_by_email("nobody@example.com")
    assert result is None


# ---------------------------------------------------------------------------
# get_active_user
# ---------------------------------------------------------------------------


async def test_get_active_user_returns_active(user_repo: UserRepository, make_user) -> None:
    """get_active_user returns an active user."""
    user = await make_user(email="active@example.com", is_active=True)
    found = await user_repo.get_active_user(user.id)
    assert found is not None
    assert found.id == user.id


async def test_get_active_user_returns_none_for_inactive(
    user_repo: UserRepository, make_user
) -> None:
    """get_active_user returns None for a soft-deleted user."""
    user = await make_user(email="inactive@example.com", is_active=False)
    found = await user_repo.get_active_user(user.id)
    assert found is None


async def test_get_active_user_returns_none_for_unknown(
    user_repo: UserRepository,
) -> None:
    """get_active_user returns None for an unknown UUID."""
    assert await user_repo.get_active_user(uuid4()) is None


# ---------------------------------------------------------------------------
# get_active_user_by_email
# ---------------------------------------------------------------------------


async def test_get_active_user_by_email_found(user_repo: UserRepository, make_user) -> None:
    """get_active_user_by_email returns active user by email."""
    await make_user(email="activebyemail@example.com", is_active=True)
    found = await user_repo.get_active_user_by_email("activebyemail@example.com")
    assert found is not None


async def test_get_active_user_by_email_inactive_returns_none(
    user_repo: UserRepository, make_user
) -> None:
    """get_active_user_by_email returns None for inactive user."""
    await make_user(email="inactivebyemail@example.com", is_active=False)
    found = await user_repo.get_active_user_by_email("inactivebyemail@example.com")
    assert found is None


async def test_get_active_user_by_email_not_found(user_repo: UserRepository) -> None:
    """get_active_user_by_email returns None when email does not exist."""
    assert await user_repo.get_active_user_by_email("ghost@example.com") is None


# ---------------------------------------------------------------------------
# email_exists
# ---------------------------------------------------------------------------


async def test_email_exists_true_for_existing(user_repo: UserRepository, make_user) -> None:
    """email_exists returns True for an existing active user."""
    await make_user(email="exists@example.com", is_active=True)
    assert await user_repo.email_exists("exists@example.com") is True


async def test_email_exists_false_for_unknown(user_repo: UserRepository) -> None:
    """email_exists returns False when the email is not registered."""
    assert await user_repo.email_exists("unknown@example.com") is False


async def test_email_exists_false_for_inactive(user_repo: UserRepository, make_user) -> None:
    """email_exists ignores inactive (soft-deleted) users."""
    await make_user(email="deluser@example.com", is_active=False)
    assert await user_repo.email_exists("deluser@example.com") is False


async def test_email_exists_excludes_given_user(user_repo: UserRepository, make_user) -> None:
    """email_exists with exclude_user_id excludes the given user from the check."""
    user = await make_user(email="selfcheck@example.com", is_active=True)
    assert await user_repo.email_exists("selfcheck@example.com", exclude_user_id=user.id) is False


# ---------------------------------------------------------------------------
# update_user
# ---------------------------------------------------------------------------


async def test_update_user_changes_fields(user_repo: UserRepository, make_user) -> None:
    """update_user modifies the specified fields and updates updated_at."""
    user = await make_user(email="update@example.com", display_name="Before")
    original_updated_at = user.updated_at

    updated = await user_repo.update_user(
        user.id,
        display_name="After",
        email="updated@example.com",
    )
    assert updated is not None
    assert updated.display_name == "After"
    assert updated.email == "updated@example.com"
    assert updated.updated_at >= original_updated_at


async def test_update_user_returns_none_for_unknown(
    user_repo: UserRepository,
) -> None:
    """update_user returns None when user does not exist."""
    result = await user_repo.update_user(uuid4(), display_name="Ghost")
    assert result is None


# ---------------------------------------------------------------------------
# soft_delete_user (deactivate)
# ---------------------------------------------------------------------------


async def test_soft_delete_user_sets_inactive(user_repo: UserRepository, make_user) -> None:
    """soft_delete_user sets is_active to False."""
    user = await make_user(email="todelete@example.com", is_active=True)
    result = await user_repo.soft_delete_user(user.id)
    assert result is not None
    assert result.is_active is False


async def test_soft_delete_user_idempotent(user_repo: UserRepository, make_user) -> None:
    """soft_delete_user on already-inactive user returns updated record (idempotent)."""
    user = await make_user(email="alreadyinactive@example.com", is_active=False)
    result = await user_repo.soft_delete_user(user.id)
    assert result is not None
    assert result.is_active is False


async def test_soft_delete_user_unknown_returns_none(
    user_repo: UserRepository,
) -> None:
    """soft_delete_user returns None for unknown user."""
    assert await user_repo.soft_delete_user(uuid4()) is None


# ---------------------------------------------------------------------------
# mark_email_verified
# ---------------------------------------------------------------------------


async def test_mark_email_verified_sets_timestamp(user_repo: UserRepository, make_user) -> None:
    """mark_email_verified sets email_verified_at to a timezone-aware datetime."""
    user = await make_user(email="verify@example.com")
    assert user.email_verified_at is None

    updated = await user_repo.mark_email_verified(user.id)
    assert updated is not None
    assert updated.email_verified_at is not None
    assert updated.email_verified_at.tzinfo is not None


# ---------------------------------------------------------------------------
# Refresh token operations
# ---------------------------------------------------------------------------


async def test_create_refresh_token(user_repo: UserRepository, make_user) -> None:
    """create_refresh_token persists the token."""
    user = await make_user(email="token@example.com")
    expires = datetime.now(UTC) + timedelta(days=7)
    token = await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="hash123",
        family_id=uuid4(),
        expires_at=expires,
        product_id="vex",
        user_agent="TestAgent/1.0",
        ip_address="127.0.0.1",
    )
    assert token.token_hash == "hash123"
    assert token.is_revoked is False


async def test_get_refresh_token_by_hash_found(user_repo: UserRepository, make_user) -> None:
    """get_refresh_token_by_hash returns the token by hash value."""
    user = await make_user(email="tokenget@example.com")
    expires = datetime.now(UTC) + timedelta(days=7)
    await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="myhash",
        family_id=uuid4(),
        expires_at=expires,
        product_id="vex",
    )
    found = await user_repo.get_refresh_token_by_hash("myhash")
    assert found is not None
    assert found.token_hash == "myhash"


async def test_get_refresh_token_by_hash_not_found(
    user_repo: UserRepository,
) -> None:
    """get_refresh_token_by_hash returns None for an unknown hash."""
    assert await user_repo.get_refresh_token_by_hash("nonexistent") is None


async def test_get_valid_refresh_token_returns_active(user_repo: UserRepository, make_user) -> None:
    """get_valid_refresh_token returns a valid non-revoked, non-expired token."""
    user = await make_user(email="validtoken@example.com")
    expires = datetime.now(UTC) + timedelta(days=7)
    await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="validhash",
        family_id=uuid4(),
        expires_at=expires,
        product_id="vex",
    )
    found = await user_repo.get_valid_refresh_token("validhash")
    assert found is not None


async def test_get_valid_refresh_token_returns_none_for_revoked(
    user_repo: UserRepository,
    make_user,
    db_session: AsyncSession,  # noqa: ARG001
) -> None:
    """get_valid_refresh_token returns None for a revoked token."""
    user = await make_user(email="revokedtoken@example.com")
    token_id = uuid4()
    expires = datetime.now(UTC) + timedelta(days=7)
    await user_repo.create_refresh_token(
        id=token_id,
        user_id=user.id,
        token_hash="revokedhash",
        family_id=uuid4(),
        expires_at=expires,
        product_id="vex",
    )
    await user_repo.revoke_refresh_token(token_id)
    found = await user_repo.get_valid_refresh_token("revokedhash")
    assert found is None


async def test_revoke_refresh_token_returns_true(user_repo: UserRepository, make_user) -> None:
    """revoke_refresh_token returns True when successfully revoked."""
    user = await make_user(email="revoke@example.com")
    token_id = uuid4()
    await user_repo.create_refresh_token(
        id=token_id,
        user_id=user.id,
        token_hash="revhash",
        family_id=uuid4(),
        expires_at=datetime.now(UTC) + timedelta(days=1),
        product_id="vex",
    )
    result = await user_repo.revoke_refresh_token(token_id)
    assert result is True


async def test_revoke_refresh_token_returns_false_for_unknown(
    user_repo: UserRepository,
) -> None:
    """revoke_refresh_token returns False for unknown token ID."""
    assert await user_repo.revoke_refresh_token(uuid4()) is False


async def test_revoke_token_family(user_repo: UserRepository, make_user) -> None:
    """revoke_token_family revokes all tokens in the same family."""
    user = await make_user(email="family@example.com")
    family_id = uuid4()
    expires = datetime.now(UTC) + timedelta(days=7)
    for i in range(3):
        await user_repo.create_refresh_token(
            id=uuid4(),
            user_id=user.id,
            token_hash=f"familyhash{i}",
            family_id=family_id,
            expires_at=expires,
            product_id="vex",
        )
    count = await user_repo.revoke_token_family(family_id)
    assert count == 3


async def test_revoke_all_user_tokens(user_repo: UserRepository, make_user) -> None:
    """revoke_all_user_tokens revokes all tokens for a user."""
    user = await make_user(email="revokeall@example.com")
    expires = datetime.now(UTC) + timedelta(days=7)
    for i in range(2):
        await user_repo.create_refresh_token(
            id=uuid4(),
            user_id=user.id,
            token_hash=f"allhash{i}",
            family_id=uuid4(),
            expires_at=expires,
            product_id="vex",
        )
    count = await user_repo.revoke_all_user_tokens(user.id)
    assert count == 2


async def test_revoke_all_user_tokens_no_tokens(user_repo: UserRepository, make_user) -> None:
    """revoke_all_user_tokens with no tokens returns 0 without error."""
    user = await make_user(email="notokens@example.com")
    count = await user_repo.revoke_all_user_tokens(user.id)
    assert count == 0


async def test_cleanup_expired_tokens_deletes_expired(user_repo: UserRepository, make_user) -> None:
    """cleanup_expired_tokens deletes expired tokens and returns the count."""
    user = await make_user(email="cleanup@example.com")
    past = datetime.now(UTC) - timedelta(hours=1)
    future = datetime.now(UTC) + timedelta(days=7)

    # 2 expired, 1 valid
    for i in range(2):
        await user_repo.create_refresh_token(
            id=uuid4(),
            user_id=user.id,
            token_hash=f"expiredhash{i}",
            family_id=uuid4(),
            expires_at=past,
            product_id="vex",
        )
    await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="validhashclean",
        family_id=uuid4(),
        expires_at=future,
        product_id="vex",
    )

    deleted = await user_repo.cleanup_expired_tokens()
    assert deleted >= 2


async def test_cleanup_expired_tokens_no_expired_returns_zero(
    user_repo: UserRepository, make_user
) -> None:
    """cleanup_expired_tokens returns 0 when there are no expired tokens."""
    user = await make_user(email="noclean@example.com")
    await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="futurehash",
        family_id=uuid4(),
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )
    deleted = await user_repo.cleanup_expired_tokens()
    assert deleted == 0


# ---------------------------------------------------------------------------
# Preferred billing account
# ---------------------------------------------------------------------------


async def test_set_and_get_preferred_billing_account(user_repo: UserRepository, make_user) -> None:
    """set_preferred_billing_account and get_preferred_billing_account round-trip."""
    user = await make_user(email="billing-pref@example.com")
    updated = await user_repo.set_preferred_billing_account(user.id, "personal")
    assert updated is not None
    pref = await user_repo.get_preferred_billing_account(user.id)
    assert pref == "personal"


async def test_get_preferred_billing_account_default_none(
    user_repo: UserRepository, make_user
) -> None:
    """get_preferred_billing_account returns None for a fresh user."""
    user = await make_user(email="nopref@example.com")
    pref = await user_repo.get_preferred_billing_account(user.id)
    assert pref is None


# ---------------------------------------------------------------------------
# User statistics
# ---------------------------------------------------------------------------


async def test_get_user_job_count_empty(user_repo: UserRepository, make_user) -> None:
    """get_user_job_count returns zeros for a user with no jobs."""
    user = await make_user(email="nojobs@example.com")
    counts = await user_repo.get_user_job_count(user.id)
    assert counts == {"total": 0, "completed": 0, "failed": 0}


async def test_get_user_output_count_empty(user_repo: UserRepository, make_user) -> None:
    """get_user_output_count returns 0 for a user with no outputs."""
    user = await make_user(email="nooutputs@example.com")
    assert await user_repo.get_user_output_count(user.id) == 0


async def test_get_user_upload_count_empty(user_repo: UserRepository, make_user) -> None:
    """get_user_upload_count returns 0 for a user with no uploads."""
    user = await make_user(email="noupload@example.com")
    assert await user_repo.get_user_upload_count(user.id) == 0


async def test_get_user_storage_bytes_empty(user_repo: UserRepository, make_user) -> None:
    """get_user_storage_bytes returns 0 for a user with no stored files."""
    user = await make_user(email="nostorage@example.com")
    assert await user_repo.get_user_storage_bytes(user.id) == 0


# ---------------------------------------------------------------------------
# cascade delete: user → refresh tokens
# ---------------------------------------------------------------------------


async def test_cascade_delete_user_removes_refresh_tokens(
    db_session: AsyncSession, make_user, user_repo: UserRepository
) -> None:
    """Deleting a User cascades to its RefreshToken rows."""
    user = await make_user(email="cascade@example.com")
    await user_repo.create_refresh_token(
        id=uuid4(),
        user_id=user.id,
        token_hash="cascadehash",
        family_id=uuid4(),
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id="vex",
    )

    await db_session.delete(user)
    await db_session.flush()

    result = await db_session.execute(select(RefreshToken).where(RefreshToken.user_id == user.id))
    assert result.scalars().all() == []


# ---------------------------------------------------------------------------
# list_users
# ---------------------------------------------------------------------------


class TestListUsers:
    async def test_list_users_returns_all_active_by_default(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """list_users returns active non-SYSTEM users when no filters applied."""
        user = User(
            id=uuid4(),
            email=f"listall-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        users, total = await user_repo.list_users()
        assert total >= 1
        assert any(u.id == user.id for u in users)

    async def test_list_users_filters_by_is_active_false(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """list_users returns only inactive users when is_active=False."""
        user = User(
            id=uuid4(),
            email=f"inactive-filter-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            is_active=False,
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        users, total = await user_repo.list_users(is_active=False)
        assert total >= 1
        assert all(not u.is_active for u in users)
        assert any(u.id == user.id for u in users)

    async def test_list_users_filters_by_role(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """list_users returns only users with the specified role."""
        admin = User(
            id=uuid4(),
            email=f"admin-role-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            role=UserRole.ADMIN,
            product_id="vex",
        )
        db_session.add(admin)
        await db_session.flush()

        users, total = await user_repo.list_users(role=UserRole.ADMIN.value)
        assert total >= 1
        assert all(u.role == UserRole.ADMIN for u in users)
        assert any(u.id == admin.id for u in users)

    async def test_list_users_filters_by_email_contains_case_insensitive(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """list_users email_contains filter is case-insensitive."""
        unique = uuid4().hex[:8]
        user = User(
            id=uuid4(),
            email=f"searchable-{unique}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        users, total = await user_repo.list_users(email_contains=unique.upper())
        assert total >= 1
        assert any(u.id == user.id for u in users)

    async def test_list_users_excludes_system_role_users(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """list_users never returns users with role=SYSTEM."""
        system = User(
            id=uuid4(),
            email=f"system-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            role=UserRole.SYSTEM,
            product_id="vex",
        )
        db_session.add(system)
        await db_session.flush()

        users, _ = await user_repo.list_users()
        assert all(u.role != UserRole.SYSTEM for u in users)
        assert all(u.id != system.id for u in users)

    async def test_list_users_pagination_limit_offset(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """list_users respects limit and offset parameters."""
        for i in range(5):
            db_session.add(
                User(
                    id=uuid4(),
                    email=f"page-{uuid4().hex[:6]}-{i}@example.com",
                    password_hash="x",
                    product_id="vex",
                )
            )
        await db_session.flush()

        page1, total = await user_repo.list_users(limit=3, offset=0)
        page2, _ = await user_repo.list_users(limit=3, offset=3)
        assert len(page1) <= 3
        assert len(page2) <= 3
        # Pages should not overlap
        page1_ids = {u.id for u in page1}
        page2_ids = {u.id for u in page2}
        assert page1_ids.isdisjoint(page2_ids)

    async def test_list_users_total_reflects_filtered_count(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """total returned by list_users matches the count of filtered users."""
        unique = uuid4().hex[:8]
        for i in range(3):
            db_session.add(
                User(
                    id=uuid4(),
                    email=f"totaltest-{unique}-{i}@example.com",
                    password_hash="x",
                    product_id="vex",
                )
            )
        await db_session.flush()

        _users, total = await user_repo.list_users(email_contains=unique)
        assert total == 3


# ---------------------------------------------------------------------------
# update_user_admin
# ---------------------------------------------------------------------------


class TestUpdateUserAdmin:
    async def test_update_user_admin_changes_role(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """update_user_admin promotes a user to admin role."""
        user = User(
            id=uuid4(),
            email=f"promote-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        updated = await user_repo.update_user_admin(user.id, role=UserRole.ADMIN.value)
        assert updated is not None
        assert updated.role == UserRole.ADMIN

    async def test_update_user_admin_changes_subscription_tier(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """update_user_admin changes subscription_tier."""
        user = User(
            id=uuid4(),
            email=f"tier-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        updated = await user_repo.update_user_admin(
            user.id, subscription_tier=SubscriptionTier.PRO.value
        )
        assert updated is not None
        assert updated.subscription_tier == SubscriptionTier.PRO

    async def test_update_user_admin_deactivates_user(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """update_user_admin sets is_active=False."""
        user = User(
            id=uuid4(),
            email=f"deactivate-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        updated = await user_repo.update_user_admin(user.id, is_active=False)
        assert updated is not None
        assert updated.is_active is False

    async def test_update_user_admin_noop_when_no_fields_given(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """update_user_admin returns the existing user without UPDATE when all args None."""
        user = User(
            id=uuid4(),
            email=f"noop-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        result = await user_repo.update_user_admin(user.id)
        assert result is not None
        assert result.id == user.id

    async def test_update_user_admin_returns_none_for_unknown_id(
        self, user_repo: UserRepository
    ) -> None:
        """update_user_admin returns None when user does not exist."""
        result = await user_repo.update_user_admin(uuid4(), role=UserRole.ADMIN.value)
        assert result is None

    async def test_update_user_admin_raises_on_system_role(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """update_user_admin raises ValueError if role=SYSTEM is requested."""
        user = User(
            id=uuid4(),
            email=f"system-guard-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()

        with pytest.raises(ValueError, match="SYSTEM"):
            await user_repo.update_user_admin(user.id, role=UserRole.SYSTEM.value)

    async def test_update_user_admin_updates_updated_at_timestamp(
        self, user_repo: UserRepository, db_session: AsyncSession
    ) -> None:
        """update_user_admin sets updated_at to a time >= the original."""
        user = User(
            id=uuid4(),
            email=f"timestamp-{uuid4().hex[:6]}@example.com",
            password_hash="x",
            product_id="vex",
        )
        db_session.add(user)
        await db_session.flush()
        original_updated_at = user.updated_at

        updated = await user_repo.update_user_admin(user.id, is_active=False)
        assert updated is not None
        assert updated.updated_at >= original_updated_at
