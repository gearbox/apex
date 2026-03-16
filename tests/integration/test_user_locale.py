"""Integration tests for user locale persistence."""

from __future__ import annotations

from src.db.repositories.user import UserRepository
from tests.integration.conftest import UserFactory


async def test_new_user_has_default_locale(
    make_user: UserFactory,
) -> None:
    """New users default to 'en' locale."""
    user = await make_user(email="locale_default@example.com")
    assert user.locale == "en"


async def test_update_user_locale(user_repo: UserRepository, make_user: UserFactory) -> None:
    """update_user with locale='sr' persists correctly."""
    user = await make_user(email="locale_update@example.com")
    updated = await user_repo.update_user(user.id, locale="sr")
    assert updated is not None
    assert updated.locale == "sr"


async def test_update_user_without_locale_leaves_unchanged(
    user_repo: UserRepository, make_user: UserFactory
) -> None:
    """update_user without locale param leaves locale unchanged."""
    user = await make_user(email="locale_noop@example.com")
    updated = await user_repo.update_user(user.id, display_name="New Name")
    assert updated is not None
    assert updated.locale == "en"  # default unchanged
