"""Integration tests for AdminNotificationRepository against a real database."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING

from src.core.enums import NotificationClass, UserRole
from src.db.repositories.admin_notifications import AdminNotificationRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from tests.integration.conftest import UserFactory


async def test_replace_preferences_is_full_set_replace(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email="prefs1@example.com", product_id="vex")
    repo = AdminNotificationRepository(db_session)

    await repo.replace_preferences(user.id, "vex", [(NotificationClass.USER_REGISTERED.value, 0)])
    prefs = await repo.list_preferences(user.id)
    assert {p.notification_class for p in prefs} == {NotificationClass.USER_REGISTERED.value}

    # Second call fully replaces the set — old class disappears.
    await repo.replace_preferences(
        user.id,
        "vex",
        [
            (NotificationClass.GENERATION_FAILED.value, 60),
            (NotificationClass.HEALTH_DEGRADED.value, 0),
        ],
    )
    prefs = await repo.list_preferences(user.id)
    assert {p.notification_class for p in prefs} == {
        NotificationClass.GENERATION_FAILED.value,
        NotificationClass.HEALTH_DEGRADED.value,
    }


async def test_replace_preferences_empty_set_clears_all(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email="prefs2@example.com", product_id="vex")
    repo = AdminNotificationRepository(db_session)

    await repo.replace_preferences(user.id, "vex", [(NotificationClass.USER_REGISTERED.value, 0)])
    await repo.replace_preferences(user.id, "vex", [])

    assert await repo.list_preferences(user.id) == []


async def test_link_token_create_confirm_and_replay(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email="link1@example.com", product_id="vex")
    repo = AdminNotificationRepository(db_session)
    now = datetime.now(UTC)

    link = await repo.upsert_link_token(user.id, "vex", "tok-1", now + timedelta(minutes=15))
    assert link.chat_id is None
    assert link.link_token == "tok-1"

    confirmed = await repo.confirm_link_by_token("tok-1", 123456789, now)
    assert confirmed is not None
    assert confirmed.chat_id == 123456789
    assert confirmed.link_token is None
    assert confirmed.token_expires_at is None
    assert confirmed.linked_at == now

    # Replay of the now-consumed token must fail — single-use.
    replayed = await repo.confirm_link_by_token("tok-1", 123456789, now)
    assert replayed is None


async def test_link_token_expired_is_rejected(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email="link2@example.com", product_id="vex")
    repo = AdminNotificationRepository(db_session)
    now = datetime.now(UTC)

    await repo.upsert_link_token(user.id, "vex", "tok-expired", now - timedelta(seconds=1))

    result = await repo.confirm_link_by_token("tok-expired", 42, now)
    assert result is None


async def test_link_token_unknown_is_rejected(db_session: AsyncSession) -> None:
    repo = AdminNotificationRepository(db_session)
    result = await repo.confirm_link_by_token("does-not-exist", 42, datetime.now(UTC))
    assert result is None


async def test_upsert_link_token_rotates_without_clearing_chat_id(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email="link3@example.com", product_id="vex")
    repo = AdminNotificationRepository(db_session)
    now = datetime.now(UTC)

    await repo.upsert_link_token(user.id, "vex", "tok-a", now + timedelta(minutes=15))
    await repo.confirm_link_by_token("tok-a", 999, now)

    # Requesting a new link while already linked rotates the token but must
    # not clear the existing chat_id/linked_at.
    rotated = await repo.upsert_link_token(user.id, "vex", "tok-b", now + timedelta(minutes=15))
    assert rotated.chat_id == 999
    assert rotated.link_token == "tok-b"


async def test_get_link_by_chat_id_hit_and_miss(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    user = await make_user(email="chatid1@example.com", product_id="vex")
    repo = AdminNotificationRepository(db_session)
    now = datetime.now(UTC)

    await repo.upsert_link_token(user.id, "vex", "tok-chatid", now + timedelta(minutes=15))
    await repo.confirm_link_by_token("tok-chatid", 987654321, now)

    found = await repo.get_link_by_chat_id(987654321)
    assert found is not None
    assert found.user_id == user.id

    assert await repo.get_link_by_chat_id(111111) is None


async def test_delete_link_removes_row(db_session: AsyncSession, make_user: UserFactory) -> None:
    user = await make_user(email="link4@example.com", product_id="vex")
    repo = AdminNotificationRepository(db_session)
    now = datetime.now(UTC)
    await repo.upsert_link_token(user.id, "vex", "tok-del", now + timedelta(minutes=15))

    deleted = await repo.delete_link(user.id)
    assert deleted is True
    assert await repo.get_link(user.id) is None

    assert await repo.delete_link(user.id) is False


async def _make_linked_subscriber(
    repo: AdminNotificationRepository,
    db_session: AsyncSession,
    make_user: UserFactory,
    *,
    email: str,
    product_id: str,
    role: UserRole,
    is_active: bool,
    notification_class: str,
    chat_id: int,
) -> None:
    user = await make_user(email=email, product_id=product_id, is_active=is_active)
    user.role = role
    await db_session.flush()

    await repo.replace_preferences(user.id, product_id, [(notification_class, 0)])

    now = datetime.now(UTC)
    await repo.upsert_link_token(user.id, product_id, f"tok-{email}", now + timedelta(minutes=15))
    if chat_id:
        await repo.confirm_link_by_token(f"tok-{email}", chat_id, now)


async def test_list_recipients_excludes_unlinked_inactive_and_non_admin(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    repo = AdminNotificationRepository(db_session)
    notification_class = NotificationClass.GENERATION_FAILED.value

    # Eligible: linked, active, admin.
    await _make_linked_subscriber(
        repo,
        db_session,
        make_user,
        email="eligible@example.com",
        product_id="vex",
        role=UserRole.ADMIN,
        is_active=True,
        notification_class=notification_class,
        chat_id=111,
    )
    # Not linked (no chat_id confirmed) — excluded.
    await _make_linked_subscriber(
        repo,
        db_session,
        make_user,
        email="unlinked@example.com",
        product_id="vex",
        role=UserRole.ADMIN,
        is_active=True,
        notification_class=notification_class,
        chat_id=0,
    )
    # Inactive admin — excluded.
    await _make_linked_subscriber(
        repo,
        db_session,
        make_user,
        email="inactive@example.com",
        product_id="vex",
        role=UserRole.ADMIN,
        is_active=False,
        notification_class=notification_class,
        chat_id=222,
    )
    # Plain user role — excluded.
    await _make_linked_subscriber(
        repo,
        db_session,
        make_user,
        email="plainuser@example.com",
        product_id="vex",
        role=UserRole.USER,
        is_active=True,
        notification_class=notification_class,
        chat_id=333,
    )

    recipients = await repo.list_recipients_for_class(notification_class)
    chat_ids = {r.chat_id for r in recipients}
    assert chat_ids == {111}


async def test_list_recipients_role_checked_at_query_time(
    db_session: AsyncSession, make_user: UserFactory
) -> None:
    """A demoted admin stops receiving without any preference cleanup."""
    repo = AdminNotificationRepository(db_session)
    notification_class = NotificationClass.USER_REGISTERED.value

    user = await make_user(email="demoted@example.com", product_id="vex", is_active=True)
    user.role = UserRole.ADMIN
    await db_session.flush()
    await repo.replace_preferences(user.id, "vex", [(notification_class, 0)])
    now = datetime.now(UTC)
    await repo.upsert_link_token(user.id, "vex", "tok-demoted", now + timedelta(minutes=15))
    await repo.confirm_link_by_token("tok-demoted", 444, now)

    recipients = await repo.list_recipients_for_class(notification_class)
    assert {r.chat_id for r in recipients} == {444}

    # Demote back to plain user — no preference row changes.
    user.role = UserRole.USER
    await db_session.flush()

    recipients = await repo.list_recipients_for_class(notification_class)
    assert recipients == []
