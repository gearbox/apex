"""K7 — SessionTerminationService.terminate_all: order, F5 reporting, never raises on Redis."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from src.api.schemas.ops_events import OpsEventType
from src.api.services.session_termination import (
    SessionTerminationResult,
    SessionTerminationService,
)

pytestmark = pytest.mark.unit


def _harness(
    *, epoch: int | None = 1_800_000_000, redis_enabled: bool = True
) -> tuple[SessionTerminationService, MagicMock]:
    """The service over mocks, plus one parent mock recording every call in order."""
    calls = MagicMock()
    user_repo = MagicMock()
    user_repo.lock_user_for_session_change = AsyncMock()
    user_repo.revoke_all_refresh_tokens = AsyncMock(return_value=3)
    token_revocation = MagicMock()
    token_revocation.enabled = redis_enabled
    token_revocation.revoke_user_sessions = AsyncMock(return_value=epoch)
    ops = MagicMock()
    ops.publish = AsyncMock()
    calls.attach_mock(user_repo.lock_user_for_session_change, "lock")
    calls.attach_mock(user_repo.revoke_all_refresh_tokens, "revoke_refresh")
    calls.attach_mock(token_revocation.revoke_user_sessions, "epoch")
    calls.attach_mock(ops.publish, "report")
    service = SessionTerminationService(
        session=MagicMock(),
        user_repo=user_repo,
        token_revocation=token_revocation,
        ops_event_bus=ops,
    )
    return service, calls


def _push_cleanup(calls: MagicMock) -> AsyncMock:
    cleanup = AsyncMock()
    calls.attach_mock(cleanup, "push")
    return cleanup


class TestTerminateAll:
    async def test_order_lock_refresh_epoch_push(self) -> None:
        service, calls = _harness()
        user_id = uuid4()
        with patch(
            "src.api.services.session_termination.delete_user_push_subscriptions",
            new=_push_cleanup(calls),
        ):
            result = await service.terminate_all(user_id, op="reset_password", source="email")

        assert result == SessionTerminationResult(revoked_refresh_tokens=3, epoch=1_800_000_000)
        assert result.bulk_access_revoked is True
        names = [c[0] for c in calls.mock_calls]
        assert names == ["lock", "revoke_refresh", "epoch", "push"]
        calls.lock.assert_awaited_once_with(user_id)
        calls.revoke_refresh.assert_awaited_once_with(user_id)
        calls.epoch.assert_awaited_once_with(user_id)
        push_kwargs = calls.push.await_args.kwargs
        assert push_kwargs == {"user_id": user_id, "op": "reset_password", "source": "email"}
        calls.report.assert_not_awaited()  # nothing to report on success

    async def test_epoch_failure_reports_but_does_not_raise(self) -> None:
        """Redis configured but the write failed → reported, result says so, no exception."""
        service, calls = _harness(epoch=None, redis_enabled=True)
        user_id = uuid4()
        with patch(
            "src.api.services.session_termination.delete_user_push_subscriptions",
            new=_push_cleanup(calls),
        ):
            result = await service.terminate_all(
                user_id, op="oauth_claim_unverified", source="oauth"
            )

        assert result.bulk_access_revoked is False
        assert result.revoked_refresh_tokens == 3
        # The order still ends with report → push cleanup (push cleanup is not skipped).
        assert [c[0] for c in calls.mock_calls] == [
            "lock",
            "revoke_refresh",
            "epoch",
            "report",
            "push",
        ]
        kwargs = calls.report.await_args.kwargs
        assert kwargs["event_type"] == OpsEventType.TOKEN_REVOCATION_FAILED
        assert kwargs["payload"].user_id == user_id
        assert kwargs["payload"].op == "oauth_claim_unverified"

    async def test_redis_unset_is_the_documented_noop_not_an_alert(self) -> None:
        service, calls = _harness(epoch=None, redis_enabled=False)
        with patch(
            "src.api.services.session_termination.delete_user_push_subscriptions",
            new=_push_cleanup(calls),
        ):
            result = await service.terminate_all(uuid4(), op="reset_password", source="email")

        assert result.bulk_access_revoked is False
        calls.report.assert_not_awaited()


class TestResultEpoch:
    """V1-g — the result exposes the written epoch; ``bulk_access_revoked`` derives from it."""

    @pytest.mark.parametrize(("epoch", "expected"), [(1_800_000_000, True), (None, False)])
    def test_v1_g_bulk_access_revoked_equals_epoch_is_not_none(
        self, epoch: int | None, expected: bool
    ) -> None:
        result = SessionTerminationResult(revoked_refresh_tokens=0, epoch=epoch)
        assert result.bulk_access_revoked is expected

    @pytest.mark.parametrize("epoch", [1_800_000_000, None])
    async def test_v1_g_terminate_all_carries_the_written_epoch(self, epoch: int | None) -> None:
        service, calls = _harness(epoch=epoch)
        with patch(
            "src.api.services.session_termination.delete_user_push_subscriptions",
            new=_push_cleanup(calls),
        ):
            result = await service.terminate_all(uuid4(), op="reset_password", source="email")
        assert result.epoch == epoch
        assert result.bulk_access_revoked is (epoch is not None)
