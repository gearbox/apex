"""Unit tests for EmailVerificationService."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.services.email_verification import (
    EmailVerificationService,
    InvalidTokenError,
    UserNotFoundError,
)
from src.api.services.token_revocation import TokenRevocationService

pytestmark = [pytest.mark.unit, pytest.mark.asyncio]


def _make_user(user_id=None, email="user@example.com", display_name="Alice", locale="en"):
    user = MagicMock()
    user.id = user_id or uuid4()
    user.email = email
    user.display_name = display_name
    user.locale = locale
    return user


def _make_svc(email_service=None, token_revocation_service=None):
    if email_service is None:
        email_service = AsyncMock()
        email_service.send_verification_email = AsyncMock()
        email_service.send_password_reset_email = AsyncMock()
    return EmailVerificationService(
        email_service=email_service,
        app_url="https://app.example.com",
        app_name="TestApp",
        token_revocation_service=(
            token_revocation_service
            if token_revocation_service is not None
            else TokenRevocationService(None, max_token_ttl_seconds=0)
        ),
    )


class TestSendVerificationEmail:
    async def test_sends_email_for_existing_user(self) -> None:
        user = _make_user()
        session = AsyncMock()
        email_mock = AsyncMock()
        svc = _make_svc(email_mock)

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
        ):
            user_repo = AsyncMock()
            user_repo.get_active_user = AsyncMock(return_value=user)
            user_repo_cls.return_value = user_repo

            token_repo = AsyncMock()
            token_repo.create_verification_token = AsyncMock(return_value="raw-token-abc")
            token_repo_cls.return_value = token_repo

            await svc.send_verification_email(user.id, session=session)

        email_mock.send_verification_email.assert_awaited_once()
        call_kwargs = email_mock.send_verification_email.call_args.kwargs
        assert "raw-token-abc" in call_kwargs["verification_url"]
        assert call_kwargs["to"] == user.email

    async def test_raises_when_user_not_found(self) -> None:
        session = AsyncMock()
        svc = _make_svc()

        with patch("src.api.services.email_verification.UserRepository") as user_repo_cls:
            user_repo = AsyncMock()
            user_repo.get_active_user = AsyncMock(return_value=None)
            user_repo_cls.return_value = user_repo

            with pytest.raises(UserNotFoundError):
                await svc.send_verification_email(uuid4(), session=session)


class TestVerifyEmail:
    async def test_verifies_valid_token(self) -> None:
        user = _make_user()
        session = AsyncMock()
        svc = _make_svc()

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
        ):
            token_repo = AsyncMock()
            token_repo.consume_verification_token = AsyncMock(return_value=user.id)
            token_repo_cls.return_value = token_repo

            user_repo = AsyncMock()
            user_repo.mark_email_verified = AsyncMock(return_value=user)
            user_repo_cls.return_value = user_repo

            result = await svc.verify_email("valid-token", session=session)

        assert result is user

    async def test_raises_invalid_token_error(self) -> None:
        session = AsyncMock()
        svc = _make_svc()

        with patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls:
            token_repo = AsyncMock()
            token_repo.consume_verification_token = AsyncMock(return_value=None)
            token_repo_cls.return_value = token_repo

            with pytest.raises(InvalidTokenError, match="expired"):
                await svc.verify_email("bad-token", session=session)

    async def test_raises_user_not_found_after_token(self) -> None:
        session = AsyncMock()
        svc = _make_svc()
        user_id = uuid4()

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
        ):
            token_repo = AsyncMock()
            token_repo.consume_verification_token = AsyncMock(return_value=user_id)
            token_repo_cls.return_value = token_repo

            user_repo = AsyncMock()
            user_repo.mark_email_verified = AsyncMock(return_value=None)
            user_repo_cls.return_value = user_repo

            with pytest.raises(UserNotFoundError):
                await svc.verify_email("token", session=session)


class TestSendPasswordResetEmail:
    async def test_sends_reset_email_when_user_found(self) -> None:
        user = _make_user()
        session = AsyncMock()
        email_mock = AsyncMock()
        svc = _make_svc(email_mock)

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
        ):
            user_repo = AsyncMock()
            user_repo.get_active_user_by_email = AsyncMock(return_value=user)
            user_repo_cls.return_value = user_repo

            token_repo = AsyncMock()
            token_repo.create_reset_token = AsyncMock(return_value="reset-xyz")
            token_repo_cls.return_value = token_repo

            await svc.send_password_reset_email(user.email, session=session, ip_address="1.2.3.4")

        email_mock.send_password_reset_email.assert_awaited_once()
        call_kwargs = email_mock.send_password_reset_email.call_args.kwargs
        assert "reset-xyz" in call_kwargs["reset_url"]

    async def test_silent_when_email_not_found(self) -> None:
        session = AsyncMock()
        email_mock = AsyncMock()
        svc = _make_svc(email_mock)

        with patch("src.api.services.email_verification.UserRepository") as user_repo_cls:
            user_repo = AsyncMock()
            user_repo.get_active_user_by_email = AsyncMock(return_value=None)
            user_repo_cls.return_value = user_repo

            # Should not raise
            await svc.send_password_reset_email("unknown@example.com", session=session)

        email_mock.send_password_reset_email.assert_not_awaited()


class TestResetPassword:
    async def test_resets_password_and_revokes_tokens(self) -> None:
        user = _make_user()
        session = AsyncMock()
        svc = _make_svc()

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
            patch("src.api.security.PasswordService") as pwd_cls,
        ):
            token_repo = AsyncMock()
            token_repo.consume_reset_token = AsyncMock(return_value=user.id)
            token_repo_cls.return_value = token_repo

            user_repo = AsyncMock()
            user_repo.update_user = AsyncMock(return_value=user)
            user_repo.revoke_all_refresh_tokens = AsyncMock(return_value=3)
            user_repo_cls.return_value = user_repo

            pwd_instance = MagicMock()
            pwd_instance.hash.return_value = "hashed_pw"
            pwd_instance.ahash = AsyncMock(return_value="hashed_pw")
            pwd_cls.return_value = pwd_instance

            result = await svc.reset_password("token", "new_password", session=session)

        assert result is user
        user_repo.revoke_all_refresh_tokens.assert_awaited_once()

    async def test_bulk_revokes_access_tokens(self) -> None:
        """R1 (issue #142) — the highest-impact bulk-revocation gap: password
        reset is the account-recovery flow a user reaches *because* they
        believe their account is compromised, so it must also kill live
        access tokens/content cookies, not just refresh tokens."""
        user = _make_user()
        session = AsyncMock()
        mock_token_revocation = AsyncMock()
        svc = _make_svc(token_revocation_service=mock_token_revocation)

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
            patch("src.api.security.PasswordService") as pwd_cls,
        ):
            token_repo = AsyncMock()
            token_repo.consume_reset_token = AsyncMock(return_value=user.id)
            token_repo_cls.return_value = token_repo

            user_repo = AsyncMock()
            user_repo.update_user = AsyncMock(return_value=user)
            user_repo.revoke_all_refresh_tokens = AsyncMock(return_value=3)
            user_repo_cls.return_value = user_repo

            pwd_instance = MagicMock()
            pwd_instance.hash.return_value = "hashed_pw"
            pwd_instance.ahash = AsyncMock(return_value="hashed_pw")
            pwd_cls.return_value = pwd_instance

            await svc.reset_password("token", "new_password", session=session)

        mock_token_revocation.revoke_user_sessions.assert_awaited_once_with(user.id)

    async def test_raises_invalid_token_on_bad_token(self) -> None:
        session = AsyncMock()
        svc = _make_svc()

        with patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls:
            token_repo = AsyncMock()
            token_repo.consume_reset_token = AsyncMock(return_value=None)
            token_repo_cls.return_value = token_repo

            with pytest.raises(InvalidTokenError, match="expired"):
                await svc.reset_password("bad-token", "pw", session=session)

    async def test_raises_user_not_found_after_token(self) -> None:
        session = AsyncMock()
        svc = _make_svc()
        user_id = uuid4()

        with (
            patch("src.api.services.email_verification.UserRepository") as user_repo_cls,
            patch("src.api.services.email_verification.AuthTokenRepository") as token_repo_cls,
            patch("src.api.security.PasswordService") as pwd_cls,
        ):
            token_repo = AsyncMock()
            token_repo.consume_reset_token = AsyncMock(return_value=user_id)
            token_repo_cls.return_value = token_repo

            user_repo = AsyncMock()
            user_repo.update_user = AsyncMock(return_value=None)
            user_repo_cls.return_value = user_repo

            pwd_instance = MagicMock()
            pwd_instance.hash.return_value = "hashed_pw"
            pwd_instance.ahash = AsyncMock(return_value="hashed_pw")
            pwd_cls.return_value = pwd_instance

            with pytest.raises(UserNotFoundError):
                await svc.reset_password("token", "pw", session=session)
