"""Unit tests for UserService (user profile management)."""

from __future__ import annotations

from datetime import UTC, date, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.age_verification import AgeVerificationError, AgeVerificationService
from src.api.services.user import (
    EmailAlreadyExistsError,
    InvalidPasswordError,
    UserNotFoundError,
    UserService,
)
from src.core.product import AgeGatePolicy, ProductConfig

pytestmark = pytest.mark.unit


def _make_product_config(policy: AgeGatePolicy = AgeGatePolicy.CHECKBOX) -> ProductConfig:
    cfg = MagicMock(spec=ProductConfig)
    cfg.age_gate = policy
    return cfg


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
    u.age_verified_at = kwargs.get("age_verified_at")
    u.date_of_birth = kwargs.get("date_of_birth")
    return u


def _make_service(
    user: MagicMock | None = None,
    password_verify: bool = True,
    age_verification_service: AgeVerificationService | None = None,
) -> tuple[UserService, AsyncMock, MagicMock]:
    repo = AsyncMock()
    pwd = MagicMock()
    pwd.verify.return_value = password_verify
    pwd.hash.return_value = "new_hash"
    pwd.averify = AsyncMock(return_value=password_verify)
    pwd.ahash = AsyncMock(return_value="new_hash")

    if user is not None:
        repo.get_user = AsyncMock(return_value=user)
    else:
        repo.get_user = AsyncMock(return_value=None)

    age_svc = age_verification_service or AgeVerificationService()
    svc = UserService(repository=repo, password_service=pwd, age_verification_service=age_svc)
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

    async def test_age_fields_in_response(self) -> None:
        ts = datetime.now(UTC)
        dob = date(1990, 6, 15)
        user = _make_user(age_verified_at=ts, date_of_birth=dob)
        svc, _, _ = _make_service(user)
        profile = await svc.get_profile(user.id)
        assert profile.age_verified is True
        assert profile.age_verified_at == ts
        assert profile.date_of_birth == dob

    async def test_age_verified_false_when_not_set(self) -> None:
        user = _make_user(age_verified_at=None)
        svc, _, _ = _make_service(user)
        profile = await svc.get_profile(user.id)
        assert profile.age_verified is False
        assert profile.age_verified_at is None


class TestUpdateProfile:
    async def test_updates_display_name(self) -> None:
        user = _make_user()
        updated = _make_user(id=user.id, email=user.email, display_name="New Name")
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=updated)

        result = await svc.update_profile(
            user.id, product_config=_make_product_config(), display_name="New Name"
        )
        assert result.display_name == "New Name"

    async def test_raises_when_user_not_found(self) -> None:
        svc, _, _ = _make_service(user=None)
        with pytest.raises(UserNotFoundError):
            await svc.update_profile(
                uuid4(), product_config=_make_product_config(), display_name="X"
            )

    async def test_raises_when_email_already_taken(self) -> None:
        user = _make_user(email="old@example.com")
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=True)

        with pytest.raises(EmailAlreadyExistsError):
            await svc.update_profile(
                user.id, product_config=_make_product_config(), email="taken@example.com"
            )

    async def test_no_email_uniqueness_check_when_email_unchanged(self) -> None:
        user = _make_user(email="same@example.com")
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=user)

        await svc.update_profile(
            user.id, product_config=_make_product_config(), email="same@example.com"
        )
        repo.email_exists.assert_not_called()

    async def test_raises_when_update_returns_none(self) -> None:
        user = _make_user()
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=None)

        with pytest.raises(UserNotFoundError):
            await svc.update_profile(
                user.id, product_config=_make_product_config(), display_name="X"
            )

    async def test_no_age_fields_leaves_age_state_untouched(self) -> None:
        """PATCH without age fields must not touch age state and must still update other fields."""
        user = _make_user(display_name="Old", age_verified_at=None)
        updated = _make_user(id=user.id, display_name="New", age_verified_at=None)
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=updated)

        result = await svc.update_profile(
            user.id, product_config=_make_product_config(), display_name="New"
        )
        assert result.display_name == "New"
        # age_verified_at was not passed to the repo
        repo.update_user.assert_awaited_once()
        call_kwargs = repo.update_user.call_args.kwargs
        assert call_kwargs["age_verified_at"] is None
        assert call_kwargs["date_of_birth"] is None

    async def test_first_age_confirmed_sets_verified_at(self) -> None:
        user = _make_user(age_verified_at=None)
        verified_user = _make_user(id=user.id, age_verified_at=datetime.now(UTC))
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=verified_user)

        result = await svc.update_profile(
            user.id,
            product_config=_make_product_config(AgeGatePolicy.CHECKBOX),
            age_confirmed=True,
        )
        assert result.age_verified is True
        call_kwargs = repo.update_user.call_args.kwargs
        assert call_kwargs["age_verified_at"] is not None

    async def test_second_confirmation_is_idempotent(self) -> None:
        """Re-confirming on an already-verified account must not update the timestamp."""
        existing_ts = datetime(2024, 1, 1, tzinfo=UTC)
        user = _make_user(age_verified_at=existing_ts)
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=user)

        await svc.update_profile(
            user.id,
            product_config=_make_product_config(AgeGatePolicy.CHECKBOX),
            age_confirmed=True,
        )
        call_kwargs = repo.update_user.call_args.kwargs
        # Must NOT pass a new timestamp — would overwrite the original
        assert call_kwargs["age_verified_at"] is None

    async def test_age_confirmed_false_raises(self) -> None:
        user = _make_user(age_verified_at=None)
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)

        with pytest.raises(AgeVerificationError, match="18 or older"):
            await svc.update_profile(
                user.id,
                product_config=_make_product_config(AgeGatePolicy.CHECKBOX),
                age_confirmed=False,
            )

    async def test_dob_write_once_same_value_is_noop(self) -> None:
        dob = date(1990, 6, 15)
        user = _make_user(date_of_birth=dob, age_verified_at=datetime.now(UTC))
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)
        repo.update_user = AsyncMock(return_value=user)

        # Same DOB → no-op, no raise
        await svc.update_profile(
            user.id,
            product_config=_make_product_config(AgeGatePolicy.DATE_OF_BIRTH),
            date_of_birth=dob,
        )
        call_kwargs = repo.update_user.call_args.kwargs
        assert call_kwargs.get("date_of_birth") is None  # same → not re-written

    async def test_dob_write_once_different_value_raises(self) -> None:
        user = _make_user(
            date_of_birth=date(1990, 6, 15),
            age_verified_at=datetime.now(UTC),
        )
        svc, repo, _ = _make_service(user)
        repo.email_exists = AsyncMock(return_value=False)

        with pytest.raises(AgeVerificationError, match="cannot be changed"):
            await svc.update_profile(
                user.id,
                product_config=_make_product_config(AgeGatePolicy.DATE_OF_BIRTH),
                date_of_birth=date(1991, 1, 1),
            )

    async def test_dob_over_18_first_time_verifies_and_stores(self) -> None:
        user = _make_user(age_verified_at=None, date_of_birth=None)
        verified_user = _make_user(
            id=user.id, age_verified_at=datetime.now(UTC), date_of_birth=date(1999, 9, 9)
        )
        svc, mock_repo, _ = _make_service(user)
        mock_repo.email_exists = AsyncMock(return_value=False)
        mock_repo.update_user = AsyncMock(return_value=verified_user)

        await svc.update_profile(
            user.id,
            product_config=_make_product_config(AgeGatePolicy.DATE_OF_BIRTH),
            date_of_birth=date(1999, 9, 9),
        )
        call_kwargs = mock_repo.update_user.call_args.kwargs
        assert call_kwargs["age_verified_at"] is not None
        assert call_kwargs["date_of_birth"] == date(1999, 9, 9)

    async def test_dob_exactly_18_today_verifies_and_stores(self) -> None:
        today = date.today()
        dob_exactly_18 = today.replace(year=today.year - 18)
        user = _make_user(age_verified_at=None, date_of_birth=None)
        verified_user = _make_user(
            id=user.id, age_verified_at=datetime.now(UTC), date_of_birth=dob_exactly_18
        )
        svc, mock_repo, _ = _make_service(user)
        mock_repo.email_exists = AsyncMock(return_value=False)
        mock_repo.update_user = AsyncMock(return_value=verified_user)

        await svc.update_profile(
            user.id,
            product_config=_make_product_config(AgeGatePolicy.DATE_OF_BIRTH),
            date_of_birth=dob_exactly_18,
        )
        call_kwargs = mock_repo.update_user.call_args.kwargs
        assert call_kwargs["age_verified_at"] is not None
        assert call_kwargs["date_of_birth"] == dob_exactly_18

    async def test_dob_under_18_rejects_and_persists_nothing(self) -> None:
        user = _make_user(age_verified_at=None, date_of_birth=None)
        svc, mock_repo, _ = _make_service(user)
        mock_repo.email_exists = AsyncMock(return_value=False)
        today = date.today()

        with pytest.raises(AgeVerificationError, match="at least 18"):
            await svc.update_profile(
                user.id,
                product_config=_make_product_config(AgeGatePolicy.DATE_OF_BIRTH),
                date_of_birth=today.replace(year=today.year - 15),
            )
        mock_repo.update_user.assert_not_called()

    async def test_dob_with_other_fields_applies_atomically(self) -> None:
        # reproduces the reported bug shape: DOB + a normal field in one request
        user = _make_user(age_verified_at=None, date_of_birth=None)
        verified_user = _make_user(
            id=user.id,
            display_name="New Name",
            age_verified_at=datetime.now(UTC),
            date_of_birth=date(1999, 9, 9),
        )
        svc, mock_repo, _ = _make_service(user)
        mock_repo.email_exists = AsyncMock(return_value=False)
        mock_repo.update_user = AsyncMock(return_value=verified_user)

        await svc.update_profile(
            user.id,
            product_config=_make_product_config(AgeGatePolicy.DATE_OF_BIRTH),
            display_name="New Name",
            date_of_birth=date(1999, 9, 9),
        )
        call_kwargs = mock_repo.update_user.call_args.kwargs
        assert call_kwargs["display_name"] == "New Name"
        assert call_kwargs["age_verified_at"] is not None
        assert call_kwargs["date_of_birth"] == date(1999, 9, 9)


class TestChangePassword:
    async def test_changes_password_and_revokes_tokens(self) -> None:
        user = _make_user()
        svc, repo, pwd = _make_service(user)
        repo.update_user = AsyncMock(return_value=user)
        repo.revoke_all_user_tokens = AsyncMock(return_value=3)

        await svc.change_password(user.id, current_password="old", new_password="new")

        pwd.averify.assert_called_once_with("hashed_pw", "old")
        pwd.ahash.assert_called_once_with("new")
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
