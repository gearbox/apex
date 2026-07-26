"""Tests for HealthCheckRegistry."""

import asyncio

import pytest

from src.api.services.health.base import ComponentHealth
from src.api.services.health.registry import HealthCheckRegistry
from src.core.enums import ComponentCategory, ComponentStatus


def _make_checker(
    name: str = "test",
    category: ComponentCategory = ComponentCategory.infrastructure,
    product_id: str | None = None,
    status: ComponentStatus = ComponentStatus.healthy,
    delay: float = 0.0,
    raise_exc: Exception | None = None,
    gates_readiness: bool = True,
) -> object:
    """Factory for test checker objects satisfying the HealthChecker protocol."""

    class _Checker:
        def __init__(self) -> None:
            self.name = name
            self.category = category
            self.product_id = product_id
            self.gates_readiness = gates_readiness

        async def check(self) -> ComponentHealth:
            if delay:
                await asyncio.sleep(delay)
            if raise_exc:
                raise raise_exc
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=status,
                latency_ms=0.0,
            )

    return _Checker()


class TestRegistry:
    def test_register_valid(self) -> None:
        reg = HealthCheckRegistry()
        checker = _make_checker()
        reg.register(checker)  # type: ignore[arg-type]
        assert len(reg.checkers) == 1

    def test_register_invalid_rejects(self) -> None:
        reg = HealthCheckRegistry()
        with pytest.raises(TypeError, match="does not satisfy"):
            reg.register("not a checker")  # type: ignore[arg-type]

    async def test_check_all_concurrent(self) -> None:
        reg = HealthCheckRegistry()
        reg.register(_make_checker(name="a"))  # type: ignore[arg-type]
        reg.register(_make_checker(name="b"))  # type: ignore[arg-type]
        results = await reg.check_all()
        assert len(results) == 2
        names = {r.name for r in results}
        assert names == {"a", "b"}

    async def test_check_all_filter_by_category(self) -> None:
        reg = HealthCheckRegistry()
        reg.register(_make_checker(name="infra", category=ComponentCategory.infrastructure))  # type: ignore[arg-type]
        reg.register(_make_checker(name="cloud", category=ComponentCategory.cloud_provider))  # type: ignore[arg-type]
        results = await reg.check_all(categories={ComponentCategory.infrastructure})
        assert len(results) == 1
        assert results[0].name == "infra"

    async def test_timeout_returns_unknown(self) -> None:
        reg = HealthCheckRegistry()
        reg.register(_make_checker(name="slow", delay=5.0))  # type: ignore[arg-type]
        results = await reg.check_all(timeout_seconds=0.1)
        assert len(results) == 1
        assert results[0].status == ComponentStatus.unknown
        assert "timed out" in results[0].message

    async def test_exception_returns_unhealthy(self) -> None:
        reg = HealthCheckRegistry()
        reg.register(_make_checker(name="broken", raise_exc=RuntimeError("db gone")))  # type: ignore[arg-type]
        results = await reg.check_all()
        assert len(results) == 1
        assert results[0].status == ComponentStatus.unhealthy
        assert "db gone" in results[0].message

    async def test_latency_measured(self) -> None:
        reg = HealthCheckRegistry()
        reg.register(_make_checker(name="fast"))  # type: ignore[arg-type]
        results = await reg.check_all()
        assert results[0].latency_ms >= 0.0

    async def test_gates_readiness_propagated_from_checker(self) -> None:
        """issue #142 G2 — the checker's gates_readiness declaration must be
        copied onto the ComponentHealth result, since HealthService.readiness()
        filters on the result, not the checker object."""
        reg = HealthCheckRegistry()
        reg.register(_make_checker(name="diagnostic-only", gates_readiness=False))  # type: ignore[arg-type]
        results = await reg.check_all()
        assert results[0].gates_readiness is False

    async def test_gates_readiness_propagated_on_timeout(self) -> None:
        reg = HealthCheckRegistry()
        reg.register(_make_checker(name="slow", delay=5.0, gates_readiness=False))  # type: ignore[arg-type]
        results = await reg.check_all(timeout_seconds=0.1)
        assert results[0].gates_readiness is False

    async def test_gates_readiness_propagated_on_exception(self) -> None:
        reg = HealthCheckRegistry()
        reg.register(
            _make_checker(name="broken", raise_exc=RuntimeError("x"), gates_readiness=False)  # type: ignore[arg-type]
        )
        results = await reg.check_all()
        assert results[0].gates_readiness is False
