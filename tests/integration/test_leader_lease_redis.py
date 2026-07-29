"""Integration test for LeaderLease against a real Redis instance (R6).

tests/unit/test_leader_lease.py exercises LeaderLease's acquire/renew/release
logic against an in-process fake client. This file re-runs the same
scenarios against a real Redis (the `redis-test` service in
docker-compose.test.yml / REDIS_URL) so the actual SET NX EX semantics and
Lua compare-and-swap scripts are proven, not just their mocked stand-ins.

Two genuinely separate `redis.asyncio.Redis` clients are used for the two
LeaderLease instances (rather than one shared client/pool) to mirror the real
deployment, where each holder is a distinct process with its own connection.

"Redis stopped" is simulated by pointing a lease at an unreachable address
rather than actually stopping the shared `redis-test` container — this suite
runs concurrently with every other integration test against that same
container, so taking it down for one test would break test isolation. An
unreachable address raises the same `redis.exceptions` classes
(ConnectionError/TimeoutError, both caught by LeaderLease's `except
(RedisError, OSError)`) that a stopped Redis would, so the fail-closed/grace
code path is exercised for real rather than mocked.
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
import pytest_asyncio
import redis.asyncio as aioredis

from src.workers.base import LeaderLease

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

_REDIS_URL = os.environ.get("REDIS_URL")

pytestmark = [
    pytest.mark.skipif(not _REDIS_URL, reason="REDIS_URL not set — no real Redis available"),
]

_LEASE_KEY = "test:leader_lease_redis:lease"

# Unroutable (TEST-NET-1, RFC 5737) address with a short connect timeout so a
# failed connection attempt raises quickly instead of hanging the test.
_UNREACHABLE_URL = "redis://192.0.2.1:6379/0"


@pytest_asyncio.fixture
async def redis_clients() -> AsyncGenerator[tuple[aioredis.Redis, aioredis.Redis]]:
    """Two genuinely separate Redis clients, simulating two processes."""
    assert _REDIS_URL is not None
    client_a = aioredis.Redis.from_url(_REDIS_URL, decode_responses=True)
    client_b = aioredis.Redis.from_url(_REDIS_URL, decode_responses=True)
    await client_a.delete(_LEASE_KEY)
    try:
        yield client_a, client_b
    finally:
        await client_a.delete(_LEASE_KEY)
        await client_a.aclose()
        await client_b.aclose()


@pytest_asyncio.fixture
async def unreachable_client() -> AsyncGenerator[aioredis.Redis]:
    client = aioredis.Redis.from_url(
        _UNREACHABLE_URL,
        decode_responses=True,
        socket_connect_timeout=0.2,
        socket_timeout=0.2,
        retry_on_timeout=False,
    )
    try:
        yield client
    finally:
        await client.aclose()


async def _acquire_via(lease: LeaderLease, client: aioredis.Redis) -> bool:
    """Drive one acquire_or_renew() call bound to a specific Redis client."""
    with patch("src.workers.base.get_redis_client", return_value=client):
        return await lease.acquire_or_renew()


class TestLeaderLeaseAgainstRealRedis:
    async def test_exactly_one_of_two_leases_acquires(
        self, redis_clients: tuple[aioredis.Redis, aioredis.Redis]
    ) -> None:
        client_a, client_b = redis_clients
        lease_a = LeaderLease(key=_LEASE_KEY, ttl_seconds=10, redis_enabled=True)
        lease_b = LeaderLease(key=_LEASE_KEY, ttl_seconds=10, redis_enabled=True)

        acquired_a = await _acquire_via(lease_a, client_a)
        acquired_b = await _acquire_via(lease_b, client_b)

        assert acquired_a is True
        assert lease_a.is_held is True
        assert acquired_b is False
        assert lease_b.is_held is False

    async def test_holder_renews_successfully(
        self, redis_clients: tuple[aioredis.Redis, aioredis.Redis]
    ) -> None:
        client_a, _client_b = redis_clients
        lease_a = LeaderLease(key=_LEASE_KEY, ttl_seconds=10, redis_enabled=True)

        assert await _acquire_via(lease_a, client_a) is True
        # Second call hits the renew (compare-and-swap EXPIRE) path, not acquire.
        assert await _acquire_via(lease_a, client_a) is True
        assert lease_a.is_held is True

    async def test_non_holder_never_promotes_itself_while_lease_is_live(
        self, redis_clients: tuple[aioredis.Redis, aioredis.Redis]
    ) -> None:
        client_a, client_b = redis_clients
        lease_a = LeaderLease(key=_LEASE_KEY, ttl_seconds=10, redis_enabled=True)
        lease_b = LeaderLease(key=_LEASE_KEY, ttl_seconds=10, redis_enabled=True)

        assert await _acquire_via(lease_a, client_a) is True

        # The non-holder retries several times against the real key held by
        # another owner token — it must never flip to held.
        for _ in range(3):
            acquired = await _acquire_via(lease_b, client_b)
            assert acquired is False
            assert lease_b.is_held is False

    async def test_holder_stays_leader_within_grace_window_when_redis_unreachable(
        self,
        redis_clients: tuple[aioredis.Redis, aioredis.Redis],
        unreachable_client: aioredis.Redis,
    ) -> None:
        client_a, _client_b = redis_clients
        # ttl=8, safety margin=5 -> ~3s grace window after the last real renewal.
        lease_a = LeaderLease(key=_LEASE_KEY, ttl_seconds=8, redis_enabled=True)

        assert await _acquire_via(lease_a, client_a) is True

        # Immediately "lose" Redis — well within the grace window.
        held = await _acquire_via(lease_a, unreachable_client)

        assert held is True
        assert lease_a.is_held is True

    async def test_holder_drops_leadership_once_grace_window_elapses(
        self,
        redis_clients: tuple[aioredis.Redis, aioredis.Redis],
        unreachable_client: aioredis.Redis,
    ) -> None:
        client_a, _client_b = redis_clients
        lease_a = LeaderLease(key=_LEASE_KEY, ttl_seconds=8, redis_enabled=True)

        assert await _acquire_via(lease_a, client_a) is True

        # Grace window is ttl_seconds - 5.0 margin ≈ 3s; wait past it before
        # the next unreachable attempt.
        await asyncio.sleep(3.5)
        held = await _acquire_via(lease_a, unreachable_client)

        assert held is False
        assert lease_a.is_held is False
