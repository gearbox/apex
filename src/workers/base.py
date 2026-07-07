"""Uniform lifecycle base for periodic background workers.

Every periodic worker loop in the app (token cleanup, GPU provisioning,
Aisha/Grok job polling, health snapshots, ...) shares the same shape: an
asyncio task ticking on an interval, an optional Redis leader lease so only
one process acts in multi-worker deployments, jittered sleeps, graceful
drain on shutdown, and a heartbeat for liveness checks.

Deliberately dependency-light: only stdlib, redis, structlog, and
``src.core``. Workers under ``src/api/services`` import this module, never
the other way around.
"""

from __future__ import annotations

import asyncio
import random
import secrets
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from datetime import UTC, datetime

import structlog

from src.core.redis import get_redis_client

logger = structlog.get_logger(__name__)

# Lua scripts kept inline (no script-cache abstraction) — each is a single
# GET-compare-then-mutate, cheap enough to send as source every call.
_RENEW_LEASE_SCRIPT = (
    "if redis.call('GET', KEYS[1]) == ARGV[1] then "
    "return redis.call('EXPIRE', KEYS[1], ARGV[2]) "
    "else return 0 end"
)
_RELEASE_LEASE_SCRIPT = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"


class LeaderLease:
    """Redis-backed owned lease with heartbeat renewal.

    Unlike a bare ``SET NX EX``, this lease carries a per-instance owner
    token: renewal and release both compare-and-swap against that token, so
    one process can never silently steal or drop a lease it doesn't own.

    Fails open on Redis errors (D5): if Redis is unreachable, the caller
    proceeds as leader. Fail-closed would turn a transient Redis outage into
    a total processing outage, which is worse than the (bounded) risk of a
    duplicate tick.
    """

    def __init__(self, *, key: str, ttl_seconds: int, redis_enabled: bool) -> None:
        self._key = key
        self._ttl_seconds = ttl_seconds
        self._enabled = redis_enabled
        self._token = secrets.token_hex(16)
        self._held = False

    @property
    def is_held(self) -> bool:
        """Whether this instance believes it currently holds the lease."""
        return self._held

    async def acquire_or_renew(self) -> bool:
        """Try to acquire the lease, or renew it if already held. Never raises."""
        if not self._enabled:
            self._held = True
            return True

        try:
            client = get_redis_client()
            acquired = await client.set(
                self._key,
                self._token,
                nx=True,
                ex=self._ttl_seconds,
            )
            if acquired:
                self._held = True
                return True

            renewed = await client.eval(
                _RENEW_LEASE_SCRIPT,
                1,
                self._key,
                self._token,
                self._ttl_seconds,
            )
            self._held = bool(renewed)
            return self._held
        except Exception:
            logger.warning("worker.lease.error", key=self._key)
            self._held = True
            return True

    async def release(self) -> None:
        """Best-effort compare-and-delete release. Never raises."""
        if not self._enabled or not self._held:
            self._held = False
            return

        try:
            client = get_redis_client()
            await client.eval(_RELEASE_LEASE_SCRIPT, 1, self._key, self._token)
        except Exception:
            logger.warning("worker.lease.release_error", key=self._key)
        finally:
            self._held = False


class PeriodicWorker(ABC):
    """Uniform lifecycle for periodic background loops.

    Subclasses implement ``run_once()``. The base owns: task management,
    leader lease, jitter, graceful drain, heartbeat, structured log events.
    """

    def __init__(
        self,
        *,
        name: str,
        interval_seconds: float,
        initial_delay_seconds: float = 0.0,
        jitter_seconds: float = 0.0,
        drain_timeout_seconds: float = 30.0,
        use_leader_lease: bool = True,
        redis_enabled: bool = False,
    ) -> None:
        self._name = name
        self._interval = interval_seconds
        self._initial_delay = initial_delay_seconds
        self._jitter = jitter_seconds
        self._drain_timeout = drain_timeout_seconds
        self._running = False
        self._task: asyncio.Task[None] | None = None
        self._started_at: datetime | None = None
        self._last_tick_at: datetime | None = None
        self._last_tick_duration_ms: int | None = None

        # TTL = 3x interval, minimum 90s — renewal happens every tick, so this
        # survives one slow tick plus one missed renewal before another
        # process could steal the lease.
        lease_ttl_seconds = max(int(interval_seconds * 3), 90)
        self._lease = LeaderLease(
            key=f"worker:{name}:lease",
            ttl_seconds=lease_ttl_seconds,
            redis_enabled=redis_enabled and use_leader_lease,
        )

    @property
    def name(self) -> str:
        return self._name

    @property
    def interval_seconds(self) -> float:
        return self._interval

    @property
    def initial_delay_seconds(self) -> float:
        return self._initial_delay

    @property
    def started_at(self) -> datetime | None:
        """UTC timestamp of the most recent start() call, or None if never started."""
        return self._started_at

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def is_leader(self) -> bool:
        """Whether this instance currently holds the leader lease.

        Updated every loop iteration. A non-leader worker is expected to be
        running but not ticking — callers (e.g. the heartbeat checker) must
        not treat that as staleness.
        """
        return self._lease.is_held

    @property
    def last_tick_at(self) -> datetime | None:
        return self._last_tick_at

    @property
    def last_tick_duration_ms(self) -> int | None:
        return self._last_tick_duration_ms

    @abstractmethod
    async def run_once(self) -> None:
        """Execute one tick of work. Raised exceptions are logged and swallowed."""
        ...

    async def start(self) -> None:
        """Start the loop in the background. Idempotent."""
        if self._running:
            return
        self._running = True
        self._started_at = datetime.now(UTC)
        self._task = asyncio.create_task(self._run_loop())
        logger.info("worker.started", name=self._name, interval_s=self._interval)

    async def stop(self) -> None:
        """Signal the loop to stop and drain the in-flight tick. Idempotent.

        ``_running`` is cleared before waiting so the loop exits at the next
        sleep boundary on its own; a hung tick is only cancelled once
        ``drain_timeout_seconds`` elapses.
        """
        if not self._running:
            return
        self._running = False

        if self._task is not None:
            task = self._task
            try:
                await asyncio.wait_for(task, timeout=self._drain_timeout)
            except TimeoutError:
                logger.warning("worker.drain_timeout", name=self._name)
                with suppress(asyncio.CancelledError):
                    await task
            self._task = None

        await self._lease.release()
        logger.info("worker.stopped", name=self._name)

    async def _run_loop(self) -> None:
        if self._initial_delay:
            await asyncio.sleep(self._initial_delay)

        while self._running:
            try:
                if await self._lease.acquire_or_renew():
                    started = time.monotonic()
                    await self.run_once()
                    self._last_tick_duration_ms = int((time.monotonic() - started) * 1000)
                    self._last_tick_at = datetime.now(UTC)
                    logger.debug(
                        "worker.tick_done",
                        name=self._name,
                        duration_ms=self._last_tick_duration_ms,
                    )
                else:
                    logger.debug("worker.tick_skipped_not_leader", name=self._name)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("worker.tick_error", name=self._name)

            if self._running:
                await asyncio.sleep(self._interval + random.uniform(0.0, self._jitter))  # noqa: S311
