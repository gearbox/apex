"""Integration tests for PushSubscriptionRepository against a real PostgreSQL database."""

from __future__ import annotations

from uuid import uuid4

from src.db.repositories.push_subscription import PushSubscriptionRepository

# ---------------------------------------------------------------------------
# upsert — insert
# ---------------------------------------------------------------------------


async def test_upsert_creates_new_subscription(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()

    subscription = await push_subscription_repo.upsert(
        user_id=user.id,
        product_id="vex",
        endpoint="https://push.example/abc",
        p256dh="p256dh-key",
        auth="auth-key",
        user_agent="test-agent",
    )

    assert subscription.user_id == user.id
    assert subscription.endpoint == "https://push.example/abc"
    assert subscription.p256dh == "p256dh-key"
    assert subscription.auth == "auth-key"
    assert subscription.user_agent == "test-agent"


async def test_get_by_endpoint_returns_created_row(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    created = await push_subscription_repo.upsert(
        user_id=user.id,
        product_id="vex",
        endpoint="https://push.example/xyz",
        p256dh="p",
        auth="a",
        user_agent=None,
    )

    found = await push_subscription_repo.get_by_endpoint("https://push.example/xyz")

    assert found is not None
    assert found.id == created.id


async def test_get_by_endpoint_returns_none_for_unknown(
    push_subscription_repo: PushSubscriptionRepository,
) -> None:
    assert await push_subscription_repo.get_by_endpoint("https://push.example/nope") is None


# ---------------------------------------------------------------------------
# upsert — reassignment across users (shared device / account switch)
# ---------------------------------------------------------------------------


async def test_upsert_reassigns_endpoint_to_new_user(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user_a = await make_user(email="a@example.com")
    user_b = await make_user(email="b@example.com")
    endpoint = "https://push.example/shared-device"

    first = await push_subscription_repo.upsert(
        user_id=user_a.id,
        product_id="vex",
        endpoint=endpoint,
        p256dh="p1",
        auth="a1",
        user_agent="device-a",
    )

    second = await push_subscription_repo.upsert(
        user_id=user_b.id,
        product_id="vex",
        endpoint=endpoint,
        p256dh="p2",
        auth="a2",
        user_agent="device-b",
    )

    # Same row (endpoint is unique), reassigned to the new user with fresh keys.
    assert second.id == first.id
    assert second.user_id == user_b.id
    assert second.p256dh == "p2"
    assert second.auth == "a2"
    assert second.user_agent == "device-b"

    # Only one row exists for this endpoint.
    all_rows = await push_subscription_repo.list_by_user(user_b.id)
    assert len(all_rows) == 1


async def test_upsert_updates_last_seen_at_on_repeat_call(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    endpoint = "https://push.example/repeat"

    first = await push_subscription_repo.upsert(
        user_id=user.id,
        product_id="vex",
        endpoint=endpoint,
        p256dh="p",
        auth="a",
        user_agent=None,
    )
    first_last_seen = first.last_seen_at

    second = await push_subscription_repo.upsert(
        user_id=user.id,
        product_id="vex",
        endpoint=endpoint,
        p256dh="p",
        auth="a",
        user_agent=None,
    )

    assert second.id == first.id
    assert second.last_seen_at >= first_last_seen


# ---------------------------------------------------------------------------
# delete_by_endpoint — ownership scoped, idempotent
# ---------------------------------------------------------------------------


async def test_delete_by_endpoint_removes_owned_subscription(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    await push_subscription_repo.upsert(
        user_id=user.id,
        product_id="vex",
        endpoint="https://push.example/to-delete",
        p256dh="p",
        auth="a",
        user_agent=None,
    )

    deleted = await push_subscription_repo.delete_by_endpoint(
        "https://push.example/to-delete", user_id=user.id
    )

    assert deleted is True
    assert await push_subscription_repo.get_by_endpoint("https://push.example/to-delete") is None


async def test_delete_by_endpoint_returns_false_when_not_found(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    deleted = await push_subscription_repo.delete_by_endpoint(
        "https://push.example/does-not-exist", user_id=user.id
    )
    assert deleted is False


async def test_delete_by_endpoint_does_not_delete_other_users_subscription(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    owner = await make_user(email="owner@example.com")
    other = await make_user(email="other@example.com")
    await push_subscription_repo.upsert(
        user_id=owner.id,
        product_id="vex",
        endpoint="https://push.example/owned",
        p256dh="p",
        auth="a",
        user_agent=None,
    )

    deleted = await push_subscription_repo.delete_by_endpoint(
        "https://push.example/owned", user_id=other.id
    )

    assert deleted is False
    assert await push_subscription_repo.get_by_endpoint("https://push.example/owned") is not None


# ---------------------------------------------------------------------------
# delete_by_id — used by PushService to prune expired subscriptions
# ---------------------------------------------------------------------------


async def test_delete_by_id_removes_subscription(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    subscription = await push_subscription_repo.upsert(
        user_id=user.id,
        product_id="vex",
        endpoint="https://push.example/prune-me",
        p256dh="p",
        auth="a",
        user_agent=None,
    )

    await push_subscription_repo.delete_by_id(subscription.id)

    assert await push_subscription_repo.get_by_endpoint("https://push.example/prune-me") is None


async def test_delete_by_id_is_a_noop_for_unknown_id(
    push_subscription_repo: PushSubscriptionRepository,
) -> None:
    # Should not raise even though nothing exists with this id.
    await push_subscription_repo.delete_by_id(uuid4())


# ---------------------------------------------------------------------------
# list_by_user
# ---------------------------------------------------------------------------


async def test_list_by_user_returns_only_that_users_subscriptions(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user_a = await make_user(email="listA@example.com")
    user_b = await make_user(email="listB@example.com")

    await push_subscription_repo.upsert(
        user_id=user_a.id,
        product_id="vex",
        endpoint="https://push.example/a1",
        p256dh="p",
        auth="a",
        user_agent=None,
    )
    await push_subscription_repo.upsert(
        user_id=user_a.id,
        product_id="vex",
        endpoint="https://push.example/a2",
        p256dh="p",
        auth="a",
        user_agent=None,
    )
    await push_subscription_repo.upsert(
        user_id=user_b.id,
        product_id="vex",
        endpoint="https://push.example/b1",
        p256dh="p",
        auth="a",
        user_agent=None,
    )

    results = await push_subscription_repo.list_by_user(user_a.id)

    assert len(results) == 2
    assert {r.endpoint for r in results} == {
        "https://push.example/a1",
        "https://push.example/a2",
    }


async def test_list_by_user_empty_for_user_with_no_subscriptions(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    assert await push_subscription_repo.list_by_user(user.id) == []


# ---------------------------------------------------------------------------
# list_batch — keyset pagination for broadcast fan-out
# ---------------------------------------------------------------------------


async def test_list_batch_paginates_by_id_ascending(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    for i in range(5):
        await push_subscription_repo.upsert(
            user_id=user.id,
            product_id="vex",
            endpoint=f"https://push.example/batch-{i}",
            p256dh="p",
            auth="a",
            user_agent=None,
        )

    first_page = await push_subscription_repo.list_batch(limit=2, cursor_id=None)
    assert len(first_page) == 2
    assert first_page[0].id < first_page[1].id

    second_page = await push_subscription_repo.list_batch(limit=2, cursor_id=first_page[-1].id)
    assert len(second_page) == 2
    assert second_page[0].id > first_page[-1].id

    # No overlap between pages.
    first_ids = {row.id for row in first_page}
    second_ids = {row.id for row in second_page}
    assert first_ids.isdisjoint(second_ids)


async def test_list_batch_returns_empty_past_the_end(
    push_subscription_repo: PushSubscriptionRepository, make_user
) -> None:
    user = await make_user()
    await push_subscription_repo.upsert(
        user_id=user.id,
        product_id="vex",
        endpoint="https://push.example/only-one",
        p256dh="p",
        auth="a",
        user_agent=None,
    )

    page = await push_subscription_repo.list_batch(limit=10, cursor_id=None)
    assert len(page) == 1

    next_page = await push_subscription_repo.list_batch(limit=10, cursor_id=page[-1].id)
    assert next_page == []
