"""Tests for HealthService."""

from src.api.services.health.base import ComponentHealth
from src.api.services.health.registry import HealthCheckRegistry
from src.api.services.health.service import HealthService
from src.core.enums import ComponentCategory, ComponentStatus


def _make_checker(
    name: str,
    category: ComponentCategory,
    status: ComponentStatus,
    product_id: str | None = None,
    gates_readiness: bool = True,
) -> object:
    class _C:
        def __init__(self) -> None:
            self.name = name
            self.category = category
            self.product_id = product_id
            self.gates_readiness = gates_readiness

        async def check(self) -> ComponentHealth:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=status,
                latency_ms=0.0,
                product_id=self.product_id,
                gates_readiness=self.gates_readiness,
            )

    return _C()


def _build_service(*checkers: object) -> HealthService:
    reg = HealthCheckRegistry()
    for c in checkers:
        reg.register(c)  # type: ignore[arg-type]
    return HealthService(registry=reg, timeout_seconds=5.0)


class TestReadiness:
    async def test_ready_when_infra_healthy(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker("redis", ComponentCategory.infrastructure, ComponentStatus.healthy),
        )
        is_ready, checks = await svc.readiness()
        assert is_ready is True
        assert checks["postgres"] == "healthy"

    async def test_not_ready_when_postgres_down(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.unhealthy),
            _make_checker("redis", ComponentCategory.infrastructure, ComponentStatus.healthy),
        )
        is_ready, _ = await svc.readiness()
        assert is_ready is False

    async def test_r2_excluded_from_readiness(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker("redis", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker(
                "r2",
                ComponentCategory.infrastructure,
                ComponentStatus.unhealthy,
                gates_readiness=False,
            ),
        )
        is_ready, checks = await svc.readiness()
        # R2 unhealthy but excluded from readiness determination
        assert is_ready is True
        # R2 should still appear in checks dict for visibility
        assert "r2" in checks

    async def test_token_revocation_inactive_does_not_fail_readiness(self) -> None:
        """issue #142 G2 — Redis unset must not permanently 503 readiness."""
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker(
                "token_revocation",
                ComponentCategory.infrastructure,
                ComponentStatus.inactive,
                gates_readiness=False,
            ),
        )
        is_ready, checks = await svc.readiness()
        assert is_ready is True
        assert checks["token_revocation"] == "inactive"

    async def test_token_revocation_breaker_open_does_not_fail_readiness(self) -> None:
        """issue #142 G2 — an open circuit breaker must not pull the whole
        API fleet out of rotation; that's the exact outage the fail-open
        posture exists to prevent."""
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker(
                "token_revocation",
                ComponentCategory.infrastructure,
                ComponentStatus.unhealthy,
                gates_readiness=False,
            ),
        )
        is_ready, checks = await svc.readiness()
        assert is_ready is True
        assert checks["token_revocation"] == "unhealthy"

    async def test_cloud_providers_excluded_from_readiness(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker(
                "grok", ComponentCategory.cloud_provider, ComponentStatus.unhealthy, "vex"
            ),
        )
        is_ready, checks = await svc.readiness()
        assert is_ready is True
        # Cloud providers not in readiness checks
        assert "grok" not in checks


class TestDetailed:
    async def test_detailed_groups_by_category(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker("redis", ComponentCategory.infrastructure, ComponentStatus.healthy),
        )
        result = await svc.detailed()
        assert result["status"] == "healthy"
        assert len(result["infrastructure"]["components"]) == 2  # type: ignore[index]
        assert result["gpu_sessions"]["status"] == "inactive"  # type: ignore[index]

    async def test_overall_status_worst_case(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker("redis", ComponentCategory.infrastructure, ComponentStatus.unhealthy),
        )
        result = await svc.detailed()
        assert result["status"] == "unhealthy"

    async def test_cloud_providers_grouped_by_product(self) -> None:
        svc = _build_service(
            _make_checker("grok", ComponentCategory.cloud_provider, ComponentStatus.healthy, "vex"),
            _make_checker(
                "grok", ComponentCategory.cloud_provider, ComponentStatus.healthy, "synthara"
            ),
        )
        result = await svc.detailed()
        assert "vex" in result["cloud_providers"]  # type: ignore[operator]
        assert "synthara" in result["cloud_providers"]  # type: ignore[operator]


class TestDetailedWithProviders:
    """Test detailed response with cloud provider and platform API data populated."""

    async def test_cloud_providers_appear_per_product(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker("grok", ComponentCategory.cloud_provider, ComponentStatus.healthy, "vex"),
            _make_checker(
                "grok", ComponentCategory.cloud_provider, ComponentStatus.healthy, "synthara"
            ),
        )
        result = await svc.detailed()
        assert "vex" in result["cloud_providers"]  # type: ignore[operator]
        assert "synthara" in result["cloud_providers"]  # type: ignore[operator]
        assert result["cloud_providers"]["vex"]["status"] == "healthy"  # type: ignore[index]

    async def test_platform_api_appears(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker("vastai_api", ComponentCategory.platform_api, ComponentStatus.healthy),
        )
        result = await svc.detailed()
        assert result["platform_apis"]["status"] == "healthy"  # type: ignore[index]
        assert len(result["platform_apis"]["components"]) == 1  # type: ignore[index]
        assert result["platform_apis"]["components"][0]["name"] == "vastai_api"  # type: ignore[index]

    async def test_inactive_platform_api_shows_inactive(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker("vastai_api", ComponentCategory.platform_api, ComponentStatus.inactive),
        )
        result = await svc.detailed()
        assert result["platform_apis"]["status"] == "inactive"  # type: ignore[index]

    async def test_unhealthy_provider_degrades_overall(self) -> None:
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker(
                "grok", ComponentCategory.cloud_provider, ComponentStatus.unhealthy, "vex"
            ),
        )
        result = await svc.detailed()
        assert result["status"] == "unhealthy"

    async def test_degraded_provider_degrades_overall(self) -> None:
        """A degraded provider (auth failure) should degrade overall status."""
        svc = _build_service(
            _make_checker("postgres", ComponentCategory.infrastructure, ComponentStatus.healthy),
            _make_checker(
                "grok", ComponentCategory.cloud_provider, ComponentStatus.degraded, "vex"
            ),
        )
        result = await svc.detailed()
        assert result["status"] == "degraded"
