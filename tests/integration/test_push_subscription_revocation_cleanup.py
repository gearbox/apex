"""Integration tests for push-subscription cleanup on bulk session revocation.

Exercises the real call chain — AuthService.logout_all / AuthService.refresh_tokens
(reuse detection) / UserService.change_password / UserService.deactivate_account /
EmailVerificationService.reset_password — against a real PostgreSQL database,
verifying that each of the five bulk-revocation sites deletes every push
subscription for the affected user (D1), that single-device logout leaves
other devices' subscriptions untouched (D2), and that a failure in push
deletion never blocks the primary action while still surfacing truthfully via
the ops event bus (D4).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.api.security import PasswordService, generate_token, hash_token
from src.api.services.auth import AuthService, TokenReuseDetectedError
from src.api.services.email_verification import EmailVerificationService
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.user import UserService
from src.core.enums import RefreshTokenRevocationReason
from src.core.uid import new_id
from src.db.models.user import RefreshToken
from src.db.repositories.push_subscription import PushSubscriptionRepository
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from tests.integration.conftest import ResetTokenFactory, UserFactory

PRODUCT_ID = "vex"


def _noop_token_revocation() -> TokenRevocationService:
    return TokenRevocationService(None, max_token_ttl_seconds=0)


async def _seed_subscriptions(
    repo: PushSubscriptionRepository, user_id: UUID, *, count: int = 2
) -> None:
    for i in range(count):
        await repo.upsert(
            user_id=user_id,
            product_id=PRODUCT_ID,
            endpoint=f"https://push.example/{user_id}-{i}",
            p256dh="p",
            auth="a",
            user_agent=None,
        )


async def _seed_refresh_token(
    session: AsyncSession,
    *,
    user_id: UUID,
    family_id: UUID | None = None,
    is_revoked: bool = False,
    revoked_reason: str | None = None,
) -> str:
    """Insert a RefreshToken row directly (flush only, no commit) and return the raw token."""
    raw_token = generate_token(32)
    token = RefreshToken(
        id=new_id(),
        user_id=user_id,
        token_hash=hash_token(raw_token),
        family_id=family_id or new_id(),
        is_revoked=is_revoked,
        revoked_reason=revoked_reason,
        expires_at=datetime.now(UTC) + timedelta(days=7),
        product_id=PRODUCT_ID,
    )
    session.add(token)
    await session.flush()
    return raw_token


class TestLogoutAllDeletesPushSubscriptions:
    async def test_two_subscriptions_end_at_zero(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        user = await make_user()
        await _seed_subscriptions(push_subscription_repo, user.id)

        auth_service = AuthService(
            repository=UserRepository(db_session),
            jwt_service=MagicMock(),
            password_service=PasswordService(),
            token_revocation_service=_noop_token_revocation(),
            session=db_session,
        )

        await auth_service.logout_all(user.id)

        assert await push_subscription_repo.list_by_user(user.id) == []


class TestChangePasswordDeletesPushSubscriptions:
    async def test_two_subscriptions_end_at_zero(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        password_service = PasswordService()
        user = await make_user(password_hash=password_service.hash("old-password"))
        await _seed_subscriptions(push_subscription_repo, user.id)

        user_service = UserService(
            repository=UserRepository(db_session),
            password_service=password_service,
            age_verification_service=MagicMock(),
            token_revocation_service=_noop_token_revocation(),
            session=db_session,
        )

        await user_service.change_password(
            user.id, current_password="old-password", new_password="new-password"
        )

        assert await push_subscription_repo.list_by_user(user.id) == []


class TestDeactivateAccountDeletesPushSubscriptions:
    async def test_two_subscriptions_end_at_zero(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        user = await make_user()
        await _seed_subscriptions(push_subscription_repo, user.id)

        user_service = UserService(
            repository=UserRepository(db_session),
            password_service=PasswordService(),
            age_verification_service=MagicMock(),
            token_revocation_service=_noop_token_revocation(),
            session=db_session,
        )

        await user_service.deactivate_account(user.id)

        assert await push_subscription_repo.list_by_user(user.id) == []


class TestResetPasswordDeletesPushSubscriptions:
    async def test_two_subscriptions_end_at_zero(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        make_reset_token: ResetTokenFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        user = await make_user()
        await _seed_subscriptions(push_subscription_repo, user.id)
        _token_row, raw_token = await make_reset_token(user=user)

        svc = EmailVerificationService(
            email_service=AsyncMock(),
            app_url="https://app.example.com",
            token_revocation_service=_noop_token_revocation(),
        )

        await svc.reset_password(raw_token, "new-password", session=db_session)

        assert await push_subscription_repo.list_by_user(user.id) == []


class TestTokenReuseDetectionDeletesPushSubscriptions:
    async def test_two_subscriptions_end_at_zero(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        user = await make_user()
        await _seed_subscriptions(push_subscription_repo, user.id)
        # Already revoked by something other than a bulk revocation (e.g. a
        # prior rotation) — replaying it must be treated as theft.
        raw_token = await _seed_refresh_token(
            db_session,
            user_id=user.id,
            is_revoked=True,
            revoked_reason=RefreshTokenRevocationReason.ROTATED.value,
        )

        auth_service = AuthService(
            repository=UserRepository(db_session),
            jwt_service=MagicMock(),
            password_service=PasswordService(),
            token_revocation_service=_noop_token_revocation(),
            session=db_session,
        )

        with pytest.raises(TokenReuseDetectedError):
            await auth_service.refresh_tokens(raw_token)

        assert await push_subscription_repo.list_by_user(user.id) == []


class TestSingleLogoutLeavesPushSubscriptionsIntact:
    """D2 — POST /v1/auth/logout ends one session; it must never touch the
    push_subscriptions table at all."""

    async def test_two_subscriptions_remain(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        user = await make_user()
        await _seed_subscriptions(push_subscription_repo, user.id)
        raw_token = await _seed_refresh_token(db_session, user_id=user.id)

        auth_service = AuthService(
            repository=UserRepository(db_session),
            jwt_service=MagicMock(),
            password_service=PasswordService(),
            token_revocation_service=_noop_token_revocation(),
            session=db_session,
        )

        result = await auth_service.logout(raw_token)

        assert result is True
        assert len(await push_subscription_repo.list_by_user(user.id)) == 2


class TestPushDeletionFailureDoesNotBlockPrimaryAction:
    """D4 — a failure in push-subscription cleanup must never block the
    primary action, and must be reported truthfully via the ops event bus."""

    async def test_logout_all_still_succeeds_and_reports_ops_event(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        user = await make_user()
        await _seed_subscriptions(push_subscription_repo, user.id)
        raw_token = await _seed_refresh_token(db_session, user_id=user.id)

        ops_event_bus = AsyncMock()
        auth_service = AuthService(
            repository=UserRepository(db_session),
            jwt_service=MagicMock(),
            password_service=PasswordService(),
            token_revocation_service=_noop_token_revocation(),
            session=db_session,
            ops_event_bus=ops_event_bus,
        )

        with patch.object(
            PushSubscriptionRepository,
            "delete_all_for_user",
            AsyncMock(side_effect=RuntimeError("db unreachable")),
        ):
            count = await auth_service.logout_all(user.id)

        # Primary action succeeded despite the push-deletion failure.
        assert count == 1

        # The SAVEPOINT rolled back only the failed delete — the outer
        # session is still healthy and the subscriptions are untouched.
        assert len(await push_subscription_repo.list_by_user(user.id)) == 2

        ops_event_bus.publish.assert_awaited_once()
        _, kwargs = ops_event_bus.publish.call_args
        assert kwargs["payload"].user_id == user.id
        assert kwargs["payload"].op == "logout_all"

        # The revoked-token write on the same session is durable.
        stored = await UserRepository(db_session).get_refresh_token_by_hash(hash_token(raw_token))
        assert stored is not None
        assert stored.is_revoked is True

    async def test_change_password_still_succeeds_and_reports_ops_event(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        password_service = PasswordService()
        user = await make_user(password_hash=password_service.hash("old-password"))
        await _seed_subscriptions(push_subscription_repo, user.id)

        ops_event_bus = AsyncMock()
        user_service = UserService(
            repository=UserRepository(db_session),
            password_service=password_service,
            age_verification_service=MagicMock(),
            token_revocation_service=_noop_token_revocation(),
            session=db_session,
            ops_event_bus=ops_event_bus,
        )

        with patch.object(
            PushSubscriptionRepository,
            "delete_all_for_user",
            AsyncMock(side_effect=RuntimeError("db unreachable")),
        ):
            await user_service.change_password(
                user.id, current_password="old-password", new_password="new-password"
            )

        assert len(await push_subscription_repo.list_by_user(user.id)) == 2
        ops_event_bus.publish.assert_awaited_once()
        _, kwargs = ops_event_bus.publish.call_args
        assert kwargs["payload"].op == "change_password"

    async def test_reset_password_still_succeeds_and_reports_ops_event(
        self,
        db_session: AsyncSession,
        make_user: UserFactory,
        make_reset_token: ResetTokenFactory,
        push_subscription_repo: PushSubscriptionRepository,
    ) -> None:
        user = await make_user()
        await _seed_subscriptions(push_subscription_repo, user.id)
        _token_row, raw_token = await make_reset_token(user=user)

        ops_event_bus = AsyncMock()
        svc = EmailVerificationService(
            email_service=AsyncMock(),
            app_url="https://app.example.com",
            token_revocation_service=_noop_token_revocation(),
            ops_event_bus=ops_event_bus,
        )

        with patch.object(
            PushSubscriptionRepository,
            "delete_all_for_user",
            AsyncMock(side_effect=RuntimeError("db unreachable")),
        ):
            await svc.reset_password(raw_token, "new-password", session=db_session)

        assert len(await push_subscription_repo.list_by_user(user.id)) == 2
        ops_event_bus.publish.assert_awaited_once()
        _, kwargs = ops_event_bus.publish.call_args
        assert kwargs["payload"].op == "reset_password"
