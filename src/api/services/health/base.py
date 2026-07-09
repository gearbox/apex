"""Health checker protocol and result types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.core.enums import ComponentCategory, ComponentStatus


@dataclass(frozen=True, kw_only=True)
class ComponentHealth:
    """Immutable result of a single health check.

    Args:
        name: Component identifier (e.g. "postgres", "redis", "grok").
        category: Taxonomy bucket.
        status: Health status.
        latency_ms: Wall-clock time of the check in milliseconds.
        message: Human-readable detail (empty string if nothing notable).
        product_id: None for global components; product slug for provider-scoped.
        metadata: Arbitrary extra data (pool stats, model counts, etc.).
    """

    name: str
    category: ComponentCategory
    status: ComponentStatus
    latency_ms: float
    message: str = ""
    product_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@runtime_checkable
class HealthChecker(Protocol):
    """Interface every health checker implements.

    Checkers are registered at startup into the HealthCheckRegistry.
    The registry calls check() concurrently and wraps each with a timeout.

    The checker must NOT measure its own latency — the registry does that.
    Return latency_ms=0.0 in ComponentHealth; the registry overwrites it.
    """

    @property
    def name(self) -> str:
        """Unique component name."""
        ...

    @property
    def category(self) -> ComponentCategory:
        """Component taxonomy."""
        ...

    @property
    def product_id(self) -> str | None:
        """None = global. Non-None = product-scoped."""
        ...

    async def check(self) -> ComponentHealth:
        """Execute the health check. Must be safe to call concurrently."""
        ...
