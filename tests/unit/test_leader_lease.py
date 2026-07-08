"""Tests for LeaderLease."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

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
        if current == token:
            return 1
        return 0


def _patched_client(client: _FakeRedisClient) -> Any:
    return patch("src.workers.base.get_redis_client", return_value=client)


class TestAcquireWhenFree:
    async def test_acquire_when_free(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)

        with _patched_client(client):
            acquired = await lease.acquire_or_renew()

        assert acquired is True
        assert lease.is_held is True


class TestSecondOwnerDenied:
    async def test_second_owner_denied_while_held(self) -> None:
        client = _FakeRedisClient()
        first = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)
        second = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)

        with _patched_client(client):
            assert await first.acquire_or_renew() is True
            assert await second.acquire_or_renew() is False

        assert second.is_held is False


class TestRenewExtendsOwnLease:
    async def test_renew_extends_own_lease(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)

        with _patched_client(client):
            assert await lease.acquire_or_renew() is True
            # Second tick: key already exists (owned by us) -> renew path.
            assert await lease.acquire_or_renew() is True

        assert lease.is_held is True


class TestRenewFailsForStolenLease:
    async def test_renew_fails_for_stolen_lease(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)
        # Someone else holds the key already.
        client._store["worker:x:lease"] = "someone-elses-token"

        with _patched_client(client):
            acquired = await lease.acquire_or_renew()

        assert acquired is False
        assert lease.is_held is False


class TestReleaseOnlyOwnLease:
    async def test_release_only_own_lease(self) -> None:
        client = _FakeRedisClient()
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)

        with _patched_client(client):
            await lease.acquire_or_renew()
            await lease.release()

        assert lease.is_held is False
        assert "worker:x:lease" not in client._store

    async def test_release_does_not_delete_other_owners_lease(self) -> None:
        client = _FakeRedisClient()
        first = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)
        second = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)

        with _patched_client(client):
            await first.acquire_or_renew()
            await second.acquire_or_renew()  # denied, second never held it
            await second.release()

        assert client._store.get("worker:x:lease") == first._token


class TestRedisErrorFailsOpen:
    async def test_redis_error_fails_open_with_warning(self) -> None:
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)

        with (
            patch("src.workers.base.get_redis_client", side_effect=ConnectionError("down")),
            patch("src.workers.base.logger") as mock_logger,
        ):
            acquired = await lease.acquire_or_renew()

        assert acquired is True
        assert lease.is_held is True
        mock_logger.warning.assert_called_once_with("worker.lease.error", key="worker:x:lease")

    async def test_release_swallows_redis_error(self) -> None:
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=True)
        lease._held = True

        with patch("src.workers.base.get_redis_client", side_effect=ConnectionError("down")):
            await lease.release()  # must not raise

        assert lease.is_held is False


class TestDisabledLeaseAlwaysLeader:
    async def test_disabled_lease_always_leader(self) -> None:
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=False)

        acquired = await lease.acquire_or_renew()

        assert acquired is True
        assert lease.is_held is True

    async def test_disabled_lease_release_is_noop(self) -> None:
        lease = LeaderLease(key="worker:x:lease", ttl_seconds=90, redis_enabled=False)
        await lease.acquire_or_renew()

        await lease.release()  # must not touch redis, must not raise

        assert lease.is_held is False
