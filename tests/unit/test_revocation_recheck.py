"""Unit tests for src.api.security.revocation_recheck.recheck_revocation_or_raise.

Exercises the ``also_lock`` locking contract in isolation, against the real
``UserRepository.lock_users_for_session_change``/``lock_user_for_session_change``
methods (only the innermost per-row lock is patched out, so the real
sort/dedup logic runs) — the deadlock-avoidance property (E2/C1) lives in
that sort, not in this helper, so these tests confirm the helper forwards
the right set rather than re-testing the repository's sorting directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import UUID, uuid4

import pytest
from litestar.exceptions import NotAuthorizedException

from src.api.security.jwt import TokenPayload
from src.api.security.revocation_recheck import recheck_revocation_or_raise
from src.db.repositories.user import UserRepository


def _payload(sub: UUID) -> TokenPayload:
    return TokenPayload(sub=str(sub), exp=9999999999, iat=0, jti="jti-1", type="access")


@pytest.fixture
def not_revoked() -> AsyncMock:
    service = AsyncMock()
    service.is_revoked.return_value = False
    return service


class TestAlsoLockForwarding:
    """also_lock must be forwarded into lock_users_for_session_change's
    sorted-set locking, which is what actually prevents the E2/C1 deadlock."""

    async def test_also_lock_empty_locks_only_the_actor(self, not_revoked: AsyncMock) -> None:
        actor_id = uuid4()
        locked: list[UUID] = []

        async def _fake_lock(self: UserRepository, user_id: UUID) -> None:  # noqa: ARG001
            locked.append(user_id)

        with patch.object(UserRepository, "lock_user_for_session_change", _fake_lock):
            await recheck_revocation_or_raise(
                session=MagicMock(),
                actor_id=actor_id,
                token_payload=_payload(actor_id),
                token_revocation_service=not_revoked,
            )

        assert locked == [actor_id]

    async def test_also_lock_target_locks_both_in_sorted_order(
        self, not_revoked: AsyncMock
    ) -> None:
        # Pick actor/target so the actor is deliberately NOT the smaller
        # UUID — proves the order comes from sorting, not argument order.
        low_id, high_id = sorted((uuid4(), uuid4()))
        actor_id, target_id = high_id, low_id
        locked: list[UUID] = []

        async def _fake_lock(self: UserRepository, user_id: UUID) -> None:  # noqa: ARG001
            locked.append(user_id)

        with patch.object(UserRepository, "lock_user_for_session_change", _fake_lock):
            await recheck_revocation_or_raise(
                session=MagicMock(),
                actor_id=actor_id,
                token_payload=_payload(actor_id),
                token_revocation_service=not_revoked,
                also_lock=(target_id,),
            )

        assert locked == [low_id, high_id]

    async def test_actor_present_in_also_lock_is_deduplicated(self, not_revoked: AsyncMock) -> None:
        actor_id = uuid4()
        locked: list[UUID] = []

        async def _fake_lock(self: UserRepository, user_id: UUID) -> None:  # noqa: ARG001
            locked.append(user_id)

        with patch.object(UserRepository, "lock_user_for_session_change", _fake_lock):
            await recheck_revocation_or_raise(
                session=MagicMock(),
                actor_id=actor_id,
                token_payload=_payload(actor_id),
                token_revocation_service=not_revoked,
                also_lock=(actor_id,),
            )

        assert locked == [actor_id]


class TestRevocationRaisesAfterLocking:
    async def test_revoked_token_raises_after_the_locks_are_taken(self) -> None:
        actor_id = uuid4()
        events: list[str] = []

        async def _fake_lock(self: UserRepository, user_id: UUID) -> None:  # noqa: ARG001
            events.append("locked")

        token_revocation_service = AsyncMock()

        async def _is_revoked(_payload: TokenPayload) -> bool:
            events.append("checked")
            return True

        token_revocation_service.is_revoked.side_effect = _is_revoked

        with (
            patch.object(UserRepository, "lock_user_for_session_change", _fake_lock),
            pytest.raises(NotAuthorizedException),
        ):
            await recheck_revocation_or_raise(
                session=MagicMock(),
                actor_id=actor_id,
                token_payload=_payload(actor_id),
                token_revocation_service=token_revocation_service,
            )

        assert events == ["locked", "checked"], (
            "the lock must be acquired before the revocation check runs, "
            "even on the path that ends up raising"
        )

    async def test_not_revoked_token_does_not_raise(self, not_revoked: AsyncMock) -> None:
        actor_id = uuid4()

        async def _fake_lock(self: UserRepository, user_id: UUID) -> None:  # noqa: ARG001
            return None

        with patch.object(UserRepository, "lock_user_for_session_change", _fake_lock):
            await recheck_revocation_or_raise(
                session=MagicMock(),
                actor_id=actor_id,
                token_payload=_payload(actor_id),
                token_revocation_service=not_revoked,
            )

        not_revoked.is_revoked.assert_awaited_once()
