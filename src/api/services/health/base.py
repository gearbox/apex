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
        gates_readiness: Copied from the checker (issue #142 G2) — whether
            this result counts toward GET /health/ready. False for
            diagnostic-only checkers (r2, token_revocation) whose degraded
            state must not pull an instance out of rotation.
        metadata: Arbitrary extra data (pool stats, model counts, etc.).
    """

    name: str
    category: ComponentCategory
    status: ComponentStatus
    latency_ms: float
    message: str = ""
    product_id: str | None = None
    gates_readiness: bool = True
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

    @property
    def gates_readiness(self) -> bool:
        """Whether this checker's status affects GET /health/ready (issue #142 G2).

        True means an unhealthy/inactive result here fails the readiness
        probe and pulls the instance out of the load balancer. Declare this
        False on a checker whose entire purpose is diagnostic — r2 (a slow
        HeadBucket shouldn't stop traffic) and token_revocation (its status
        quo IS fail-open; failing readiness on it would reintroduce the
        total-outage scenario the fail-open posture exists to avoid).
        Declared on the checker itself, at the point of definition, rather
        than in a name set living somewhere else — new checkers can't
        forget to opt out.
        """
        ...

    async def check(self) -> ComponentHealth:
        """Execute the health check. Must be safe to call concurrently."""
        ...
