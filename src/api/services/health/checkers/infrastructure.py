"""Infrastructure health checkers — Postgres, Redis, R2."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import text

from src.api.services.health.base import ComponentHealth
from src.core.enums import ComponentCategory, ComponentStatus

if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.storage import R2StorageService

logger = structlog.get_logger()


class PostgresChecker:
    """Check PostgreSQL connectivity via SELECT 1."""

    name = "postgres"
    category = ComponentCategory.infrastructure
    product_id = None
    gates_readiness = True

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def check(self) -> ComponentHealth:
        # This fallback covers a failure while entering the session context.
        # On the normal path, reset the timer immediately before the query so
        # the persisted latency represents only the database round trip.
        start = time.perf_counter()
        try:
            async with self._session_factory() as session:
                start = time.perf_counter()
                await session.execute(text("SELECT 1"))
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.healthy,
                latency_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as exc:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=str(exc),
            )


class RedisChecker:
    """Check Redis connectivity via PING."""

    name = "redis"
    category = ComponentCategory.infrastructure
    product_id = None
    gates_readiness = True

    def __init__(self, redis: Redis) -> None:
        self._redis = redis

    async def check(self) -> ComponentHealth:
        start = time.perf_counter()
        try:
            result = await self._redis.ping()
            if result:
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.healthy,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=f"unexpected PING response: {result!r}",
            )
        except Exception as exc:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=str(exc),
            )


class R2Checker:
    """Check Cloudflare R2 (S3-compatible) connectivity via HeadBucket.

    Delegates to R2StorageService.health_check() which manages the
    aioboto3 context manager internally.
    """

    name = "r2"
    category = ComponentCategory.infrastructure
    product_id = None
    # issue #142 G2 — HeadBucket can be slow; not needed for accepting
    # traffic, so a degraded R2 must not fail the readiness probe.
    gates_readiness = False

    def __init__(self, r2_storage: R2StorageService) -> None:
        self._r2_storage = r2_storage

    async def check(self) -> ComponentHealth:
        start = time.perf_counter()
        try:
            ok = await self._r2_storage.health_check()
            if ok:
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.healthy,
                    latency_ms=(time.perf_counter() - start) * 1000,
                )
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=(time.perf_counter() - start) * 1000,
                message="HeadBucket check failed",
            )
        except Exception as exc:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=(time.perf_counter() - start) * 1000,
                message=str(exc),
            )
