"""Tests for LeaderLease."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError

from src.workers.base import LeaderLease


class _FakeRedisClient:
    """Minimal stateful double implementing the two ops LeaderLease needs.

    Real Lua-script semantics for SET NX EX / the renew / release scripts,
    backed by a plain dict — enough to exercise LeaderLease's actual
    acquire/renew/release logic instead of just asserting mock call args.
    """

    def __init__(self) -> None:
        self._store: dict[str, str] = {}

    async def set(self, key: str, value: str, *, nx: bool = False, ex: int | None = None) -> bool:
        del ex
        if nx and key in self._store:
            return False
        self._store[key] = value
        return True

    async def eval(self, script: str, _numkeys: int, key: str, *args: Any) -> int:
        current = self._store.get(key)
        token = args[0]
        if "DEL" in script:
            if current == token:
                del self._store[key]
                return 1
            return 0
        # renew script
        return 1 if current == token else 0


def _client_factory(client: _FakeRedisClient) -> Any:
    return lambda: client


class TestAcquireWhenFree:
    async def test_acquire_when_free(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        acquired = await lease.acquire_or_renew()

        assert acquired is True
        assert lease.is_held is True


class TestSecondOwnerDenied:
    async def test_second_owner_denied_while_held(self) -> None:
        client = _FakeRedisClient()
        first = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )
        second = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        assert await first.acquire_or_renew() is True
        assert await second.acquire_or_renew() is False

        assert second.is_held is False


class TestRenewExtendsOwnLease:
    async def test_renew_extends_own_lease(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        assert await lease.acquire_or_renew() is True
        # Second tick: key already exists (owned by us) -> renew path.
        assert await lease.acquire_or_renew() is True

        assert lease.is_held is True


class TestLeaseRenewalTimingLog:
    async def test_logs_duration_on_successful_acquire(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        with patch("src.workers.base.logger") as mock_logger:
            assert await lease.acquire_or_renew() is True

        mock_logger.debug.assert_called_once()
        assert mock_logger.debug.call_args.args == ("worker.lease.renewed",)
        assert mock_logger.debug.call_args.kwargs["key"] == "worker:x:lease"
        assert isinstance(mock_logger.debug.call_args.kwargs["duration_ms"], float)

    async def test_logs_duration_on_successful_renew(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )
        assert await lease.acquire_or_renew() is True

        with patch("src.workers.base.logger") as mock_logger:
            assert await lease.acquire_or_renew() is True

        mock_logger.debug.assert_called_once()
        assert mock_logger.debug.call_args.args == ("worker.lease.renewed",)
        assert mock_logger.debug.call_args.kwargs["key"] == "worker:x:lease"
        assert isinstance(mock_logger.debug.call_args.kwargs["duration_ms"], float)


class TestRenewFailsForStolenLease:
    async def test_renew_fails_for_stolen_lease(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )
        # Someone else holds the key already.
        client._store["worker:x:lease"] = "someone-elses-token"

        acquired = await lease.acquire_or_renew()

        assert acquired is False
        assert lease.is_held is False


class TestReleaseOnlyOwnLease:
    async def test_release_only_own_lease(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        await lease.acquire_or_renew()
        await lease.release()

        assert lease.is_held is False
        assert "worker:x:lease" not in client._store

    async def test_release_does_not_delete_other_owners_lease(self) -> None:
        client = _FakeRedisClient()
        first = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )
        second = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        await first.acquire_or_renew()
        await second.acquire_or_renew()  # denied, second never held it
        await second.release()

        assert client._store.get("worker:x:lease") == first._token


class TestRedisErrorFailsClosedWhenNotHeld:
    async def test_redis_error_without_held_lease_fails_closed(self) -> None:
        """A process that never held the lease must never promote itself on error.

        Uses redis.exceptions.ConnectionError (a RedisError, NOT a builtin
        OSError) to exercise the RedisError arm of the narrowed except clause
        specifically — the grace-window tests below cover the OSError arm.
        """
        client_factory = MagicMock(side_effect=RedisConnectionError("down"))
        lease = LeaderLease(
            key="worker:x:lease", ttl_seconds=90, redis_enabled=True, client_factory=client_factory
        )

        with (
            patch("src.workers.base.logger") as mock_logger,
        ):
            acquired = await lease.acquire_or_renew()

        assert acquired is False
        assert lease.is_held is False
        mock_logger.warning.assert_called_once_with(
            "worker.lease.error",
            key="worker:x:lease",
            held=False,
            within_grace=False,
            error="down",
            error_type="ConnectionError",
        )

    async def test_type_error_propagates_instead_of_forfeiting_leadership(self) -> None:
        """A programming error must surface as a tick failure, not "not leader".

        Only genuine Redis/transport failures (RedisError, OSError) are
        allowed to be swallowed and converted into a fail-closed result.
        """
        client_factory = MagicMock(side_effect=TypeError("boom"))
        lease = LeaderLease(
            key="worker:x:lease", ttl_seconds=90, redis_enabled=True, client_factory=client_factory
        )

        with (
            pytest.raises(TypeError, match="boom"),
        ):
            await lease.acquire_or_renew()

        assert lease.is_held is False


class TestRedisErrorWithinGraceWindow:
    async def test_redis_error_within_grace_stays_leader(self) -> None:
        """A recently-renewed key is still provably live in Redis — safe to stay leader."""
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        assert await lease.acquire_or_renew() is True

        assert lease._last_renew_at is not None
        with (
            patch("src.workers.base.time.monotonic", return_value=lease._last_renew_at + 10.0),
            patch.object(lease, "_client_factory", side_effect=ConnectionError("down")),
            patch("src.workers.base.logger") as mock_logger,
        ):
            acquired = await lease.acquire_or_renew()

        assert acquired is True
        assert lease.is_held is True
        mock_logger.warning.assert_called_once_with(
            "worker.lease.error",
            key="worker:x:lease",
            held=True,
            within_grace=True,
            error="down",
            error_type="ConnectionError",
        )

    async def test_redis_error_past_grace_fails_closed(self) -> None:
        """Elapsed time past ttl - margin means the key may already be gone."""
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        assert await lease.acquire_or_renew() is True

        assert lease._last_renew_at is not None
        with (
            patch("src.workers.base.time.monotonic", return_value=lease._last_renew_at + 90.0),
            patch.object(lease, "_client_factory", side_effect=ConnectionError("down")),
            patch("src.workers.base.logger") as mock_logger,
        ):
            acquired = await lease.acquire_or_renew()

        assert acquired is False
        assert lease.is_held is False
        mock_logger.warning.assert_called_once_with(
            "worker.lease.error",
            key="worker:x:lease",
            held=True,
            within_grace=False,
            error="down",
            error_type="ConnectionError",
        )

    async def test_consecutive_out_of_grace_misses_escalate_to_error(self) -> None:
        """A lease that cannot be renewed at all is an infrastructure fault, not a warning."""
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        assert await lease.acquire_or_renew() is True

        assert lease._last_renew_at is not None
        with (
            patch("src.workers.base.time.monotonic", return_value=lease._last_renew_at + 90.0),
            patch.object(lease, "_client_factory", side_effect=ConnectionError("down")),
            patch("src.workers.base.logger") as mock_logger,
        ):
            assert await lease.acquire_or_renew() is False  # 1st miss: warning
            assert await lease.acquire_or_renew() is False  # 2nd consecutive miss: error

        mock_logger.warning.assert_called_once_with(
            "worker.lease.error",
            key="worker:x:lease",
            held=True,
            within_grace=False,
            error="down",
            error_type="ConnectionError",
        )
        mock_logger.error.assert_called_once_with(
            "worker.lease.error",
            key="worker:x:lease",
            held=False,
            within_grace=False,
            error="down",
            error_type="ConnectionError",
        )

    async def test_recovery_after_grace_failure_resets_state(self) -> None:
        """A successful renew after an out-of-grace failure clears the miss streak.

        The failed renewal never actually deleted the Redis key (we only lost
        the ability to confirm it), so once Redis is reachable again the same
        token still owns it and the renew script succeeds.
        """
        client = _FakeRedisClient()
        lease = LeaderLease(
            key="worker:x:lease",
            ttl_seconds=90,
            redis_enabled=True,
            client_factory=_client_factory(client),
        )

        assert await lease.acquire_or_renew() is True

        assert lease._last_renew_at is not None
        with (
            patch("src.workers.base.time.monotonic", return_value=lease._last_renew_at + 90.0),
            patch.object(lease, "_client_factory", side_effect=ConnectionError("down")),
        ):
            assert await lease.acquire_or_renew() is False
            assert lease._consecutive_grace_misses == 1

        assert await lease.acquire_or_renew() is True

        assert lease._consecutive_grace_misses == 0


class TestReleaseSwallowsRedisError:
    async def test_release_swallows_redis_error(self) -> None:
        client_factory = MagicMock(side_effect=ConnectionError("down"))
        lease = LeaderLease(
            key="worker:x:lease", ttl_seconds=90, redis_enabled=True, client_factory=client_factory
        )
        lease._held = True

        with (
            patch("src.workers.base.logger") as mock_logger,
        ):
            await lease.release()  # must not raise

        assert lease.is_held is False
        assert lease._last_renew_at is None
        mock_logger.warning.assert_called_once_with(
            "worker.lease.release_error",
            key="worker:x:lease",
            error="down",
            error_type="ConnectionError",
        )

    async def test_release_type_error_propagates(self) -> None:
        """A programming error in release() must surface too, not be swallowed."""
        client_factory = MagicMock(side_effect=TypeError("boom"))
        lease = LeaderLease(
            key="worker:x:lease", ttl_seconds=90, redis_enabled=True, client_factory=client_factory
        )
        lease._held = True

        with (
            pytest.raises(TypeError, match="boom"),
        ):
            await lease.release()


class TestDisabledLeaseAlwaysLeader:
    async def test_disabled_lease_always_leader(self) -> None:
        lease = LeaderLease(
            key="worker:x:lease", ttl_seconds=90, redis_enabled=False, client_factory=MagicMock()
        )

        acquired = await lease.acquire_or_renew()

        assert acquired is True
        assert lease.is_held is True

    async def test_disabled_lease_release_is_noop(self) -> None:
        lease = LeaderLease(
            key="worker:x:lease", ttl_seconds=90, redis_enabled=False, client_factory=MagicMock()
        )
        await lease.acquire_or_renew()

        await lease.release()  # must not touch redis, must not raise

        assert lease.is_held is False
