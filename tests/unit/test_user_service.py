"""Unit tests for UserService (user profile management)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.user import (
    EmailAlreadyExistsError,
    InvalidPasswordError,
    UserNotFoundError,
    UserService,
)

pytestmark = pytest.mark.unit


def _make_user(**kwargs: object) -> MagicMock:
    u = MagicMock()
    u.id = kwargs.get("id", uuid4())
    u.email = kwargs.get("email", "test@example.com")
    u.display_name = kwargs.get("display_name", "Test User")
    u.subscription_tier = "free"
    u.locale = "en"
    u.role = "user"
    u.is_active = kwargs.get("is_active", True)
    u.password_hash = "hashed_pw"
    u.created_at = datetime.now(UTC)
    u.updated_at = datetime.now(UTC)
    return u


def _make_service(
    user: MagicMock | None = None,
    password_verify: bool = True,
) -> tuple[UserService, AsyncMock, MagicMock]:
    repo = AsyncMock()
    pwd = MagicMock()
    pwd.verify.return_value = password_verify
    pwd.hash.return_value = "new_hash"

    if user is not None:
        repo.get_user = AsyncMock(return_value=user)
    else:
        repo.get_user = AsyncMock(return_value=None)

    svc = UserService(repository=repo, password_service=pwd)
    return svc, repo, pwd


class TestGetProfile:
    async def test_returns_profile_for_existing_user(self) -> None:
        user = _make_user()
        svc, _, _ = _make_service(user)
        profile = await svc.get_profile(user.id)
        assert profile.email == user.email
        assert profile.id == str(user.id)

    async def test_raises_when_user_not_found(self) -> None:
        svc, _, _ = _make_service(user=None)
        with pytest.raises(UserNotFoundError):
            await svc.get_profile(uuid4())


class TestUpdateProfile:
    async def test_updates_display_name(self) -> None:
        user = _make_user()
        updated = _make_user(id=user.id, email=user.email, display_name="New Name")
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=updated)

        result = await svc.update_profile(user.id, display_name="New Name")
        assert result.display_name == "New Name"

    async def test_raises_when_user_not_found(self) -> None:
        svc, _, _ = _make_service(user=None)
        with pytest.raises(UserNotFoundError):
            await svc.update_profile(uuid4(), display_name="X")

    async def test_raises_when_email_already_taken(self) -> None:
        user = _make_user(email="old@example.com")
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=True)

        with pytest.raises(EmailAlreadyExistsError):
            await svc.update_profile(user.id, email="taken@example.com")

    async def test_no_email_uniqueness_check_when_email_unchanged(self) -> None:
        user = _make_user(email="same@example.com")
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=user)

        # Same email (lowercased comparison) — should NOT check email_exists
        await svc.update_profile(user.id, email="same@example.com")
        repo.email_exists.assert_not_called()

    async def test_raises_when_update_returns_none(self) -> None:
        user = _make_user()
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=None)

        with pytest.raises(UserNotFoundError):
            await svc.update_profile(user.id, display_name="X")


class TestChangePassword:
    async def test_changes_password_and_revokes_tokens(self) -> None:
        user = _make_user()
        svc, repo, pwd = _make_service(user)
        repo.update_user = AsyncMock(return_value=user)
        repo.revoke_all_user_tokens = AsyncMock(return_value=3)

        await svc.change_password(user.id, current_password="old", new_password="new")

        pwd.verify.assert_called_once_with("hashed_pw", "old")
        pwd.hash.assert_called_once_with("new")
        repo.revoke_all_user_tokens.assert_awaited_once_with(user.id)

    async def test_raises_when_user_not_found(self) -> None:
        svc, _, _ = _make_service(user=None)
        with pytest.raises(UserNotFoundError):
            await svc.change_password(uuid4(), current_password="a", new_password="b")

    async def test_raises_when_password_wrong(self) -> None:
        user = _make_user()
        svc, _, _ = _make_service(user, password_verify=False)
        with pytest.raises(InvalidPasswordError):
            await svc.change_password(user.id, current_password="wrong", new_password="new")


class TestDeactivateAccount:
    async def test_deactivates_and_returns_timestamp(self) -> None:
        user = _make_user()
        svc, repo, _ = _make_service(user)
        repo.soft_delete_user = AsyncMock(return_value=user)
        repo.revoke_all_user_tokens = AsyncMock(return_value=1)

        ts = await svc.deactivate_account(user.id)
        assert isinstance(ts, datetime)
        repo.revoke_all_user_tokens.assert_awaited_once_with(user.id)

    async def test_raises_when_user_not_found(self) -> None:
        svc, repo, _ = _make_service(user=None)
        repo.soft_delete_user = AsyncMock(return_value=None)

        with pytest.raises(UserNotFoundError):
            await svc.deactivate_account(uuid4())


class TestGetStats:
    async def test_returns_stats_for_existing_user(self) -> None:
        user = _make_user()
        svc, repo, _ = _make_service(user)
        repo.get_user_job_count = AsyncMock(return_value={"total": 10, "completed": 8, "failed": 2})
        repo.get_user_output_count = AsyncMock(return_value=15)
        repo.get_user_upload_count = AsyncMock(return_value=5)
        repo.get_user_storage_bytes = AsyncMock(return_value=1024)

        stats = await svc.get_stats(user.id)
        assert stats.total_jobs == 10
        assert stats.completed_jobs == 8
        assert stats.failed_jobs == 2
        assert stats.total_outputs == 15
        assert stats.total_uploads == 5
        assert stats.storage_used_bytes == 1024

    async def test_raises_when_user_not_found(self) -> None:
        svc, _, _ = _make_service(user=None)
        with pytest.raises(UserNotFoundError):
            await svc.get_stats(uuid4())
