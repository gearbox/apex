"""Infrastructure health checkers — Postgres, Redis, R2."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog
from sqlalchemy import text

from src.core.enums import ComponentCategory, ComponentStatus

from ..base import ComponentHealth

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.storage import R2StorageService

logger = structlog.get_logger()


class PostgresChecker:
    """Check PostgreSQL connectivity via SELECT 1."""

    name = "postgres"
    category = ComponentCategory.infrastructure
    product_id = None

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def check(self) -> ComponentHealth:
        try:
            async with self._session_factory() as session:
                await session.execute(text("SELECT 1"))
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.healthy,
                latency_ms=0.0,
            )
        except Exception as exc:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
                message=str(exc),
            )


class RedisChecker:
    """Check Redis connectivity via PING."""

    name = "redis"
    category = ComponentCategory.infrastructure
    product_id = None

    def __init__(self, redis: Any) -> None:
        self._redis = redis

    async def check(self) -> ComponentHealth:
        try:
            result = await self._redis.ping()
            if result is True or result == b"PONG":
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.healthy,
                    latency_ms=0.0,
                )
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
                message=f"unexpected PING response: {result!r}",
            )
        except Exception as exc:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
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

    def __init__(self, r2_storage: R2StorageService) -> None:
        self._r2_storage = r2_storage

    async def check(self) -> ComponentHealth:
        try:
            ok = await self._r2_storage.health_check()
            if ok:
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.healthy,
                    latency_ms=0.0,
                )
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
                message="HeadBucket check failed",
            )
        except Exception as exc:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.unhealthy,
                latency_ms=0.0,
                message=str(exc),
            )
