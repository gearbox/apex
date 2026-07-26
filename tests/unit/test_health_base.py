"""Tests for health check base types."""

import pytest

from src.api.services.health.base import ComponentHealth, HealthChecker
from src.core.enums import ComponentCategory, ComponentStatus


class TestComponentHealth:
    def test_frozen_immutable(self) -> None:
        h = ComponentHealth(
            name="test",
            category=ComponentCategory.infrastructure,
            status=ComponentStatus.healthy,
            latency_ms=1.5,
        )
        with pytest.raises(AttributeError):
            h.status = ComponentStatus.unhealthy  # type: ignore[misc]

    def test_defaults(self) -> None:
        h = ComponentHealth(
            name="x",
            category=ComponentCategory.infrastructure,
            status=ComponentStatus.healthy,
            latency_ms=0.0,
        )
        assert h.message == ""
        assert h.product_id is None
        assert h.metadata == {}

    def test_protocol_compliance(self) -> None:
        """A class with matching attrs/methods is a HealthChecker."""

        class FakeChecker:
            name = "fake"
            category = ComponentCategory.infrastructure
            product_id = None
            gates_readiness = True

            async def check(self) -> ComponentHealth:
                return ComponentHealth(
                    name=self.name,
                    category=self.category,
                    status=ComponentStatus.healthy,
                    latency_ms=0.0,
                )

        assert isinstance(FakeChecker(), HealthChecker)

    def test_non_compliant_rejects(self) -> None:
        """A class missing check() is not a HealthChecker."""

        class NotAChecker:
            name = "bad"

        assert not isinstance(NotAChecker(), HealthChecker)
