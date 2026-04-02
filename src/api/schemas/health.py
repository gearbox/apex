"""Health check response schemas (msgspec Structs)."""

import msgspec


class ComponentHealthResponse(msgspec.Struct, kw_only=True):
    """Single component status in the detailed response."""

    name: str
    status: str
    latency_ms: float
    message: str = ""
    metadata: dict[str, object] = {}


class CategoryHealthResponse(msgspec.Struct, kw_only=True):
    """Aggregate status of a category (infrastructure, platform_apis, etc.)."""

    status: str
    components: list[ComponentHealthResponse]


class GpuSessionHealthResponse(msgspec.Struct, kw_only=True):
    """GPU session reconciliation summary."""

    status: str
    total: int
    healthy: int
    stale: int
    message: str = ""


class DetailedHealthResponse(msgspec.Struct, kw_only=True):
    """Full admin health response — GET /v1/admin/health."""

    status: str
    checked_at: str
    infrastructure: CategoryHealthResponse
    platform_apis: CategoryHealthResponse
    cloud_providers: dict[str, CategoryHealthResponse]
    gpu_sessions: GpuSessionHealthResponse


class LivenessResponse(msgspec.Struct, kw_only=True):
    """Liveness probe response — GET /health/live."""

    status: str


class ReadinessResponse(msgspec.Struct, kw_only=True):
    """Readiness probe response — GET /health/ready."""

    status: str
    checks: dict[str, str]


class HealthSnapshotResponse(msgspec.Struct, kw_only=True):
    """Historical snapshot for dashboard charts — GET /v1/admin/health/history."""

    checked_at: str  # ISO 8601
    overall_status: str  # healthy | degraded | unhealthy
    snapshot_data: dict  # type: ignore[type-arg]  # Full DetailedHealthResponse
