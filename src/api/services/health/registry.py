"""Health check registry — collects checkers and runs them concurrently."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

import structlog

from src.core.enums import ComponentCategory, ComponentStatus

from .base import ComponentHealth, HealthChecker

logger = structlog.get_logger()


class HealthCheckRegistry:
    """Registry of all health checkers. Populated at app startup."""

    def __init__(self) -> None:
        self._checkers: list[HealthChecker] = []

    def register(self, checker: HealthChecker) -> None:
        """Register a health checker.

        Args:
            checker: Must satisfy the HealthChecker protocol.

        Raises:
            TypeError: If checker doesn't satisfy the protocol.
        """
        if not isinstance(checker, HealthChecker):
            msg = f"{type(checker).__name__} does not satisfy HealthChecker protocol"
            raise TypeError(msg)
        logger.info("health.checker.registered", name=checker.name, category=checker.category.value)
        self._checkers.append(checker)

    @property
    def checkers(self) -> Sequence[HealthChecker]:
        """Read-only view of registered checkers."""
        return self._checkers

    async def check_all(
        self,
        *,
        timeout_seconds: float = 5.0,
        categories: set[ComponentCategory] | None = None,
    ) -> list[ComponentHealth]:
        """Run all (or filtered) checkers concurrently with per-check timeout.

        Args:
            timeout_seconds: Maximum time for each individual check.
            categories: If provided, only run checkers in these categories.

        Returns:
            List of ComponentHealth results (one per checker).
        """
        selected = self._checkers
        if categories is not None:
            selected = [c for c in self._checkers if c.category in categories]

        if not selected:
            return []

        tasks = [self._run_single(checker, timeout_seconds) for checker in selected]
        return list(await asyncio.gather(*tasks))

    async def _run_single(
        self,
        checker: HealthChecker,
        timeout_seconds: float,
    ) -> ComponentHealth:
        """Run a single checker with timeout and latency measurement.

        Never raises — returns unhealthy/unknown on failure.
        """
        start = time.monotonic()
        try:
            result = await asyncio.wait_for(
                checker.check(),
                timeout=timeout_seconds,
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            # Overwrite latency_ms with actual measured value
            return ComponentHealth(
                name=result.name,
                category=result.category,
                status=result.status,
                latency_ms=round(elapsed_ms, 2),
                message=result.message,
                product_id=result.product_id,
                metadata=result.metadata,
            )
        except TimeoutError:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.warning("health.check.timeout", name=checker.name, timeout_s=timeout_seconds)
            return ComponentHealth(
                name=checker.name,
                category=checker.category,
                status=ComponentStatus.unknown,
                latency_ms=round(elapsed_ms, 2),
                message=f"check timed out after {timeout_seconds}s",
                product_id=checker.product_id,
            )
        except Exception as exc:
            elapsed_ms = (time.monotonic() - start) * 1000
            logger.exception("health.check.error", name=checker.name)
            return ComponentHealth(
                name=checker.name,
                category=checker.category,
                status=ComponentStatus.unhealthy,
                latency_ms=round(elapsed_ms, 2),
                message=str(exc),
                product_id=checker.product_id,
            )
