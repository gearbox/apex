# Health Check System — Design Document

## 1. Overview

Replace the current single-endpoint ComfyUI health check with a comprehensive, extensible health monitoring system covering infrastructure components, cloud AI providers, and on-demand GPU session lifecycle.

### Design Goals

- **Three-tier HTTP surface**: liveness → readiness → admin-detailed
- **Extensible provider taxonomy**: cloud-based, on-demand (Vast.ai nodes), platform APIs (Vast.ai itself)
- **Historical persistence**: configurable-interval DB snapshots for uptime/latency dashboards
- **Real-time streaming**: SSE channel for the admin panel
- **Session staleness detection**: reconcile active GPU sessions against actual node reachability, flag stale sessions for future auto-recovery
- **Global infra + per-product provider scoping**: infrastructure health is system-wide; provider availability is product-scoped (vex may enable different providers than synthara)

---

## 2. Component Taxonomy

Every checkable component implements a single async `check()` interface. Components are classified into three categories that determine expected-state semantics:

### 2.1 Infrastructure (always expected up)

| Component    | Check Method                                 | Degraded Meaning         |
|-------------|----------------------------------------------|--------------------------|
| `postgres`  | `SELECT 1` via async session                  | Fatal — no DB access     |
| `redis`     | `PING` command                                | Degraded — rate limits, SSE, caching broken |
| `r2`        | `HeadBucket` on the configured bucket         | Degraded — content storage unavailable |
| `api`       | Always healthy (if you're reading this, it's up) | N/A                     |

### 2.2 Cloud Providers (always-on external APIs)

| Provider     | Primary Check                    | Fallback Check           | Product Scope |
|-------------|----------------------------------|--------------------------|---------------|
| `grok`      | xAI status/health endpoint       | `GET /v1/models` (auth)  | vex, synthara |
| `veo3`      | (future) Google API status       | List models              | synthara      |
| `sora`      | (future) OpenAI status           | List models              | vex, synthara |
| `nano banana`      | (future) Provider status         | Auth check               | TBD           |

Strategy: prefer a dedicated status/health endpoint; fall back to a lightweight authenticated endpoint (e.g. list-models) if none exists. Each provider checker declares both.

### 2.3 Platform APIs (infrastructure for on-demand providers)

| Component      | Check Method                          | Degraded Meaning                    |
|---------------|---------------------------------------|-------------------------------------|
| `vastai_api`  | `GET /api/v1/` or auth-check endpoint | Cannot deploy/manage GPU nodes      |

### 2.4 On-Demand Providers (Aisha / GPU Sessions)

Not checked as individual "providers" in the traditional sense. Instead, the health system performs **session reconciliation**:

- Query all `gpu_sessions` with `status = 'active'`
- For each, probe the ComfyUI `/object_info` endpoint on the session's `node_host:node_port`
- Sessions where the probe fails are flagged `stale` with a `stale_detected_at` timestamp
- Future: a recovery worker picks up stale sessions and redeploys

This is **not** per-node monitoring — it's a lifecycle integrity check. Ephemeral nodes don't need uptime charts; what matters is "does reality match what our DB claims."

---

## 3. Architecture

### 3.1 Core Abstractions

```
src/api/services/health/
├── __init__.py
├── base.py              # HealthChecker protocol + dataclasses
├── registry.py          # HealthCheckRegistry — collects all checkers
├── service.py           # HealthService — orchestrates checks, snapshot persistence
├── checkers/
│   ├── __init__.py
│   ├── infrastructure.py  # PostgresChecker, RedisChecker, R2Checker
│   ├── cloud_provider.py  # CloudProviderChecker (generic), GrokChecker, etc.
│   ├── platform_api.py    # VastAIChecker
│   └── gpu_session.py     # GpuSessionReconciler
└── worker.py            # Background snapshot worker (asyncio.Task)
```

### 3.2 The `HealthChecker` Protocol

```python
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class ComponentStatus(str, enum.Enum):
    """Health status of a single component."""
    healthy = "healthy"
    degraded = "degraded"
    unhealthy = "unhealthy"
    unknown = "unknown"          # check timed out or raised unexpectedly
    inactive = "inactive"        # expected to be off (e.g. no active GPU sessions)


class ComponentCategory(str, enum.Enum):
    """Taxonomy bucket for a health component."""
    infrastructure = "infrastructure"
    cloud_provider = "cloud_provider"
    platform_api = "platform_api"
    gpu_session = "gpu_session"


@dataclass(frozen=True, kw_only=True)
class ComponentHealth:
    """Result of a single health check."""
    name: str
    category: ComponentCategory
    status: ComponentStatus
    latency_ms: float              # wall-clock time of the check
    message: str = ""              # human-readable detail
    product_id: str | None = None  # None = global (infrastructure)
    metadata: dict[str, object] = field(default_factory=dict)
    # metadata examples:
    #   postgres: {"pool_size": 10, "pool_checked_out": 3}
    #   grok: {"models_available": 4}
    #   gpu_session: {"session_id": "...", "stale": true}


@runtime_checkable
class HealthChecker(Protocol):
    """Interface every health checker implements."""

    @property
    def name(self) -> str: ...

    @property
    def category(self) -> ComponentCategory: ...

    @property
    def product_id(self) -> str | None:
        """None = global component. Non-None = product-scoped provider."""
        ...

    async def check(self) -> ComponentHealth: ...
```

**Key design decisions:**

- `Protocol` with `runtime_checkable` rather than an ABC — no forced inheritance, clean structural subtyping
- `frozen=True` dataclasses — health results are immutable value objects, safe to cache and serialize
- `latency_ms` captured by the caller (the registry orchestrator), not self-reported — prevents cheating and keeps checkers simple
- `product_id` on the result enables the "global infra + per-product providers" split at the serialization layer

### 3.3 `HealthCheckRegistry`

A simple collector that groups checkers by category:

```python
class HealthCheckRegistry:
    """Registry of all health checkers. Populated at app startup."""

    def __init__(self) -> None:
        self._checkers: list[HealthChecker] = []

    def register(self, checker: HealthChecker) -> None:
        self._checkers.append(checker)

    async def check_all(
        self,
        *,
        timeout_seconds: float = 5.0,
        categories: set[ComponentCategory] | None = None,
    ) -> list[ComponentHealth]:
        """Run all (or filtered) checkers concurrently with per-check timeout."""
        ...
```

- All checks run via `asyncio.gather` with individual `asyncio.wait_for(timeout)` wrapping
- Timed-out checks return `ComponentStatus.unknown`
- Unhandled exceptions → `ComponentStatus.unhealthy` with exception message

### 3.4 `HealthService`

The business-logic orchestrator, injected via Litestar DI:

```python
class HealthService:
    def __init__(
        self,
        registry: HealthCheckRegistry,
        session_factory: async_sessionmaker[AsyncSession],
        event_bus: EventBus,
    ) -> None: ...

    async def liveness(self) -> bool:
        """Trivially True — the process is alive."""
        return True

    async def readiness(self) -> bool:
        """Infra-only subset: postgres + redis. Returns False if any unhealthy."""
        results = await self._registry.check_all(
            categories={ComponentCategory.infrastructure},
            timeout_seconds=3.0,
        )
        return all(r.status == ComponentStatus.healthy for r in results)

    async def detailed(self) -> DetailedHealthResponse:
        """Full system check — all categories. Admin-only."""
        results = await self._registry.check_all()
        return self._build_response(results)

    async def persist_snapshot(self, results: list[ComponentHealth]) -> None:
        """Write a health snapshot row to DB for historical dashboards."""
        ...

    async def publish_realtime(self, results: list[ComponentHealth]) -> None:
        """Publish to EventBus for SSE consumers."""
        ...
```

### 3.5 Checker Implementations (representative)

#### PostgresChecker

```python
class PostgresChecker:
    name = "postgres"
    category = ComponentCategory.infrastructure
    product_id = None  # global

    def __init__(self, session_factory: async_sessionmaker[AsyncSession]) -> None:
        self._session_factory = session_factory

    async def check(self) -> ComponentHealth:
        async with self._session_factory() as session:
            await session.execute(text("SELECT 1"))
        # pool stats from engine if available
        return ComponentHealth(
            name=self.name,
            category=self.category,
            status=ComponentStatus.healthy,
            latency_ms=0,  # filled by registry wrapper
        )
```

#### GrokChecker (cloud provider pattern)

```python
class GrokChecker:
    name = "grok"
    category = ComponentCategory.cloud_provider

    def __init__(self, http_client: httpx.AsyncClient, api_key: str, product_id: str) -> None:
        self._client = http_client
        self._api_key = api_key
        self._product_id = product_id

    @property
    def product_id(self) -> str:
        return self._product_id

    async def check(self) -> ComponentHealth:
        # Strategy: prefer dedicated status endpoint, fallback to list-models
        for url in (self._status_url, self._models_url):
            try:
                resp = await self._client.get(
                    url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    timeout=5.0,
                )
                if resp.status_code < 500:
                    return ComponentHealth(
                        name=self.name,
                        category=self.category,
                        status=ComponentStatus.healthy,
                        latency_ms=0,
                        product_id=self._product_id,
                    )
            except httpx.RequestError:
                continue

        return ComponentHealth(
            name=self.name,
            category=self.category,
            status=ComponentStatus.unhealthy,
            latency_ms=0,
            message="all probe endpoints unreachable",
            product_id=self._product_id,
        )
```

#### GpuSessionReconciler

This is **not** a standard checker — it returns *multiple* `ComponentHealth` results (one per active session) plus an aggregate. It gets special handling in the registry:

```python
class GpuSessionReconciler:
    """Reconcile DB-claimed active GPU sessions against actual node reachability."""

    name = "gpu_sessions"
    category = ComponentCategory.gpu_session
    product_id = None  # operates across all products

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        http_client: httpx.AsyncClient,
    ) -> None:
        self._session_factory = session_factory
        self._client = http_client

    async def check(self) -> ComponentHealth:
        """Returns aggregate status. Marks stale sessions in DB as side-effect."""
        active_sessions = await self._get_active_sessions()

        if not active_sessions:
            return ComponentHealth(
                name=self.name,
                category=self.category,
                status=ComponentStatus.inactive,
                latency_ms=0,
                message="no active gpu sessions",
                metadata={"total": 0, "healthy": 0, "stale": 0},
            )

        healthy, stale = 0, 0
        stale_session_ids: list[UUID] = []

        probe_tasks = [
            self._probe_node(s.node_host, s.node_port)
            for s in active_sessions
        ]
        results = await asyncio.gather(*probe_tasks, return_exceptions=True)

        for session, result in zip(active_sessions, results):
            if result is True:
                healthy += 1
            else:
                stale += 1
                stale_session_ids.append(session.id)

        # Mark stale sessions in DB (set stale_detected_at if not already set)
        if stale_session_ids:
            await self._mark_stale(stale_session_ids)

        total = len(active_sessions)
        status = (
            ComponentStatus.healthy if stale == 0
            else ComponentStatus.degraded if healthy > 0
            else ComponentStatus.unhealthy
        )

        return ComponentHealth(
            name=self.name,
            category=self.category,
            status=status,
            latency_ms=0,
            message=f"{healthy}/{total} nodes reachable, {stale} stale",
            metadata={"total": total, "healthy": healthy, "stale": stale},
        )

    async def _probe_node(self, host: str, port: int) -> bool:
        """Probe ComfyUI /object_info on a GPU node."""
        try:
            resp = await self._client.get(
                f"http://{host}:{port}/object_info",
                timeout=10.0,
            )
            return resp.status_code == 200
        except httpx.RequestError:
            return False
```

### 3.6 Checker Registration at Startup

In `src/api/dependencies/common.py` → `init_services()`:

```python
# Health check registry
health_registry = HealthCheckRegistry()

# Infrastructure (global)
health_registry.register(PostgresChecker(session_factory=db_manager.session_factory))
health_registry.register(RedisChecker(redis=redis_client))
health_registry.register(R2Checker(r2_client=r2_storage.client, bucket=settings.r2_bucket_name))

# Platform APIs (global)
health_registry.register(VastAIChecker(http_client=http_client, api_key=settings.vastai_api_key))

# Cloud providers (per-product — register one checker per product that uses the provider)
for product_id, product_config in product_configs.items():
    if product_config.has_provider("grok"):
        health_registry.register(
            GrokChecker(
                http_client=http_client,
                api_key=settings.xai_api_key,
                product_id=product_id,
            )
        )

# GPU session reconciler (global — operates across products)
health_registry.register(GpuSessionReconciler(
    session_factory=db_manager.session_factory,
    http_client=http_client,
))

# Health service
health_service = HealthService(
    registry=health_registry,
    session_factory=db_manager.session_factory,
    event_bus=event_bus,
)
```

Adding a new provider = writing one `XyzChecker` class + one `register()` call. No other code changes.

---

## 4. HTTP Endpoints

### 4.1 Tier 1 — Liveness (`GET /health/live`)

Used by Docker `HEALTHCHECK`, load balancers, Kubernetes probes.

```
Response: 200 OK
Body: {"status": "alive"}
```

No auth. No DB. No external calls. Infallible — if the process serves HTTP, it's alive.

### 4.2 Tier 2 — Readiness (`GET /health/ready`)

Used by load balancers to decide if the instance should receive traffic.

```
Response: 200 OK | 503 Service Unavailable
Body: {"status": "ready" | "not_ready", "checks": {"postgres": "healthy", "redis": "healthy"}}
```

No auth. Only checks infrastructure components with a tight timeout (3s). Does **not** probe external providers — those being down shouldn't remove us from the load balancer.

### 4.3 Tier 3 — Detailed Admin (`GET /v1/admin/health`)

Full system status. Admin-only (`admin_guard`).

```json
{
  "status": "degraded",
  "checked_at": "2026-03-31T14:22:00Z",
  "infrastructure": {
    "status": "healthy",
    "components": [
      {
        "name": "postgres",
        "status": "healthy",
        "latency_ms": 1.2,
        "metadata": {"pool_size": 10, "pool_checked_out": 2}
      },
      {
        "name": "redis",
        "status": "healthy",
        "latency_ms": 0.4
      },
      {
        "name": "r2",
        "status": "healthy",
        "latency_ms": 45.3
      }
    ]
  },
  "platform_apis": {
    "status": "healthy",
    "components": [
      {
        "name": "vastai_api",
        "status": "healthy",
        "latency_ms": 120.5
      }
    ]
  },
  "cloud_providers": {
    "vex": {
      "status": "healthy",
      "components": [
        {"name": "grok", "status": "healthy", "latency_ms": 89.2}
      ]
    },
    "synthara": {
      "status": "healthy",
      "components": [
        {"name": "grok", "status": "healthy", "latency_ms": 91.0}
      ]
    }
  },
  "gpu_sessions": {
    "status": "inactive",
    "total": 0,
    "healthy": 0,
    "stale": 0,
    "message": "no active gpu sessions"
  }
}
```

Top-level `status` is the worst status across all categories — if any component is `unhealthy`, the aggregate is `unhealthy`; if any is `degraded` but none `unhealthy`, aggregate is `degraded`.

### 4.4 SSE Stream (`GET /v1/admin/health/stream`)

Admin-only. Streams `health.snapshot` events at the configured interval (same data as the detailed endpoint). The frontend admin panel subscribes here for real-time updates:

```
event: health.snapshot
data: { ... same JSON as detailed endpoint ... }
```

Uses the existing `EventBus` over Redis Pub/Sub. Channel: `health:stream`.

---

## 5. Response DTOs (msgspec)

```python
import msgspec


class ComponentHealthResponse(msgspec.Struct, kw_only=True):
    name: str
    status: str              # ComponentStatus.value
    latency_ms: float
    message: str = ""
    metadata: dict[str, object] = {}


class CategoryHealthResponse(msgspec.Struct, kw_only=True):
    status: str
    components: list[ComponentHealthResponse]


class GpuSessionHealthResponse(msgspec.Struct, kw_only=True):
    status: str
    total: int
    healthy: int
    stale: int
    message: str = ""


class DetailedHealthResponse(msgspec.Struct, kw_only=True):
    status: str
    checked_at: str          # ISO 8601
    infrastructure: CategoryHealthResponse
    platform_apis: CategoryHealthResponse
    cloud_providers: dict[str, CategoryHealthResponse]  # keyed by product_id
    gpu_sessions: GpuSessionHealthResponse


class LivenessResponse(msgspec.Struct, kw_only=True):
    status: str              # "alive"


class ReadinessResponse(msgspec.Struct, kw_only=True):
    status: str              # "ready" | "not_ready"
    checks: dict[str, str]   # component_name → status value
```

---

## 6. Data Model — Health Snapshots

### 6.1 `health_snapshots` Table

```sql
CREATE TABLE health_snapshots (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    overall_status  VARCHAR(16) NOT NULL,       -- healthy | degraded | unhealthy
    snapshot_data   JSONB NOT NULL,             -- full DetailedHealthResponse
    created_at      TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Index for dashboard time-range queries
CREATE INDEX ix_health_snapshots_checked_at ON health_snapshots (checked_at DESC);

-- Partition-friendly: retention cleanup deletes by checked_at range
```

**No `product_id` column** — the snapshot is global, but `snapshot_data` JSONB contains the per-product provider breakdown internally.

**Why JSONB and not normalized?** The snapshot schema will evolve as providers are added/removed. A normalized table (one row per component per snapshot) would require schema changes for every new provider. JSONB is append-friendly, queryable via Postgres JSON operators for dashboards, and the snapshot is always read as a whole unit anyway.

### 6.2 Retention

A periodic cleanup job (daily) deletes snapshots older than `HEALTH_SNAPSHOT_RETENTION_DAYS` (default: 30). This runs as an `asyncio.Task` alongside the snapshot worker.

### 6.3 SQLAlchemy Model

```python
class HealthSnapshot(Base):
    __tablename__ = "health_snapshots"

    id: Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()")
    )
    checked_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )
    overall_status: Mapped[str] = mapped_column(String(16), nullable=False)
    snapshot_data: Mapped[dict] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=text("CURRENT_TIMESTAMP")
    )

    __table_args__ = (
        Index("ix_health_snapshots_checked_at", "checked_at", postgresql_using="btree"),
    )
```

---

## 7. Background Snapshot Worker

An `asyncio.Task` started at app lifespan, running the check loop:

```python
class HealthSnapshotWorker:
    """Periodic health check → DB snapshot + SSE publish."""

    def __init__(
        self,
        health_service: HealthService,
        interval_seconds: float,            # from HEALTH_SNAPSHOT_INTERVAL_SECONDS env
        retention_days: int,                 # from HEALTH_SNAPSHOT_RETENTION_DAYS env
    ) -> None: ...

    async def run(self) -> None:
        """Main loop — started as asyncio.Task in app lifespan."""
        while True:
            try:
                results = await self._health_service.check_all()
                await self._health_service.persist_snapshot(results)
                await self._health_service.publish_realtime(results)
            except Exception:
                logger.exception("health.snapshot.worker.error")
            await asyncio.sleep(self._interval_seconds)

    async def cleanup_old_snapshots(self) -> None:
        """Delete snapshots older than retention period. Runs daily."""
        ...
```

### Configuration (added to `Settings`)

```python
# Health check settings
health_snapshot_interval_seconds: int = Field(default=60, ge=10, le=600)
health_snapshot_retention_days: int = Field(default=30, ge=1)
health_check_timeout_seconds: float = Field(default=5.0, ge=1.0, le=30.0)
```

---

## 8. Stale Session Detection (GPU Session Reconciliation)

The `GpuSessionReconciler` is the mechanism for detecting nodes that disappeared:

### Flow

```
┌───────────────────────────────┐
│  health snapshot tick (60s)   │
└──────────┬────────────────────┘
           │
           ▼
┌───────────────────────────────┐
│  SELECT * FROM gpu_sessions   │
│  WHERE status = 'active'      │
└──────────┬────────────────────┘
           │
           ▼  (for each session, concurrently)
┌───────────────────────────────┐
│  GET http://{host}:{port}     │
│       /object_info            │
│  timeout: 10s                 │
└──────────┬────────────────────┘
           │
    ┌──────┴──────┐
    │             │
  200 OK      timeout/error
    │             │
    ▼             ▼
  (healthy)   UPDATE gpu_sessions
              SET stale_detected_at = NOW()
              WHERE id = ? AND stale_detected_at IS NULL
```

### `gpu_sessions` Table Additions

The `gpu_sessions` table (from the GPU billing design) needs these columns for staleness tracking:

```sql
stale_detected_at   TIMESTAMPTZ,       -- NULL = not stale; set by reconciler
stale_notified      BOOLEAN DEFAULT FALSE,  -- future: admin notification sent
```

### Future: Auto-Recovery Worker

A separate worker (not part of this feature) will watch for sessions where `stale_detected_at` is older than a configurable threshold and trigger redeployment. The health check system just **detects and records** — it doesn't remediate.

---

## 9. Route Registration

```python
# src/api/routes/health.py

class HealthController(Controller):
    """Three-tier health check endpoints."""

    path = "/health"
    tags = ["health"]

    @get("/live", status_code=200)
    async def liveness(self) -> LivenessResponse:
        return LivenessResponse(status="alive")

    @get("/ready", status_code=200)
    async def readiness(
        self, health_service: HealthService,
    ) -> Response[ReadinessResponse]:
        ready, checks = await health_service.readiness()
        body = ReadinessResponse(
            status="ready" if ready else "not_ready",
            checks=checks,
        )
        return Response(
            content=body,
            status_code=200 if ready else 503,
        )


class AdminHealthController(Controller):
    """Detailed health — admin only."""

    path = "/v1/admin/health"
    tags = ["admin", "health"]
    guards = [admin_guard]

    @get("/", status_code=200)
    async def detailed(
        self, health_service: HealthService,
    ) -> DetailedHealthResponse:
        return await health_service.detailed()

    @get("/stream")
    async def stream(
        self, health_service: HealthService,
    ) -> Stream:
        """SSE stream of health snapshots."""
        ...

    @get("/history")
    async def history(
        self,
        health_service: HealthService,
        after: str | None = None,
        before: str | None = None,
        limit: int = 60,
    ) -> list[HealthSnapshotResponse]:
        """Paginated historical snapshots for dashboard charts."""
        ...
```

Note: `HealthController` has **no auth guards** — liveness and readiness are public. `AdminHealthController` is behind `admin_guard`.

---

## 10. Implementation Plan

### Phase 1 — Core Framework + Infrastructure Checks

1. **New enums** in `src/core/enums.py`: `ComponentStatus`, `ComponentCategory`
2. **Health checker protocol + dataclasses** in `src/api/services/health/base.py`
3. **HealthCheckRegistry** in `src/api/services/health/registry.py`
4. **Infrastructure checkers**: `PostgresChecker`, `RedisChecker`, `R2Checker`
5. **HealthService** (orchestrator) — `liveness()`, `readiness()`, `detailed()`
6. **Response DTOs** in `src/api/schemas/health.py`
7. **HealthController** + **AdminHealthController** routes
8. **DI wiring** in `dependencies/common.py`
9. **Tests**: unit tests for each checker (mocked deps), integration test for registry concurrency + timeout handling

**Replaces** the existing `/health/` endpoint.

### Phase 2 — Cloud Provider Checkers

1. **GrokChecker** (first real cloud provider)
2. **Generic `CloudProviderChecker` base** with status-url + fallback-url pattern
3. **VastAIChecker** (platform API)
4. Per-product registration logic
5. **Tests**: mock HTTP responses, verify fallback behavior

### Phase 3 — GPU Session Reconciler

1. **GpuSessionReconciler** checker
2. `stale_detected_at` / `stale_notified` columns on `gpu_sessions` (coordinate with GPU billing feature)
3. Concurrent node probing with timeout
4. **Tests**: mock DB sessions + HTTP probes, verify stale marking

### Phase 4 — Persistence + Real-Time

1. **`health_snapshots` table** + Alembic migration
2. **`HealthSnapshot` SQLAlchemy model** + repository
3. **HealthSnapshotWorker** background task
4. **SSE stream endpoint** via EventBus
5. **History endpoint** with time-range filtering
6. **Retention cleanup** task
7. **Configuration** env vars in Settings
8. **Tests**: worker loop, snapshot persistence, retention cleanup

### Phase 5 — Admin Panel Frontend (separate PR)

1. Real-time health dashboard subscribing to SSE
2. Uptime/latency charts from history endpoint
3. Stale session alerts

---

## 11. File Inventory

```
# New files
src/api/services/health/__init__.py
src/api/services/health/base.py             # Protocol, enums, dataclasses
src/api/services/health/registry.py          # HealthCheckRegistry
src/api/services/health/service.py           # HealthService
src/api/services/health/worker.py            # HealthSnapshotWorker
src/api/services/health/checkers/__init__.py
src/api/services/health/checkers/infrastructure.py
src/api/services/health/checkers/cloud_provider.py
src/api/services/health/checkers/platform_api.py
src/api/services/health/checkers/gpu_session.py
src/api/schemas/health.py                   # msgspec response DTOs
src/api/routes/health.py                    # HealthController + AdminHealthController
src/db/models/health.py                     # HealthSnapshot model
src/db/repositories/health.py               # HealthSnapshotRepository
tests/unit/test_health_checkers.py
tests/unit/test_health_registry.py
tests/unit/test_health_service.py
tests/integration/test_health_endpoints.py

# Modified files
src/core/enums.py                           # + ComponentStatus, ComponentCategory
src/core/config.py                          # + health_* settings
src/api/dependencies/common.py              # + health registry/service wiring
src/api/app.py                              # + worker lifecycle, route registration
alembic/versions/XXX_add_health_snapshots.py
```

---

## 12. Pitfalls & Edge Cases

### Circular dependency risk
`PostgresChecker` needs a session factory. The session factory is created during `init_services()`. Health checkers must be registered **after** core services are initialized — not at import time. The registry is populated imperatively, not via decorators.

### Check timeout vs. snapshot interval
Individual check timeout (default 5s) must be less than the snapshot interval (default 60s). If all checks time out simultaneously, worst case is `N * 5s` if run sequentially — but we use `asyncio.gather`, so it's still just 5s wall-clock. Validate: `health_check_timeout_seconds < health_snapshot_interval_seconds`.

### Granian multi-worker
The snapshot worker must run in **exactly one** worker process. Options:
- Use a Redis-based leader election lock (`SET health:worker:lock NX EX 90`) — the worker that acquires the lock runs the loop; others skip.
- Alternatively, run the worker as a separate entrypoint / sidecar container.
- **Recommended**: Redis lock — simpler, no infra changes.

### R2 HeadBucket latency
`HeadBucket` on R2 can be slow (100-300ms). This is fine for the background worker but should not block readiness checks. R2 is intentionally **excluded** from the readiness probe — it's infrastructure but not critical for accepting HTTP traffic.

### Cloud provider rate limits
Health checks hitting xAI `/v1/models` every 60s = 1440 req/day. This is well within typical API rate limits, but if it becomes an issue, the checker can cache the last-known status and only re-probe every N intervals (configurable `check_every_n_ticks`).

### Stale detection false positives
A node might be temporarily unreachable due to network blip. The reconciler sets `stale_detected_at` but does NOT take any action. The future recovery worker should require the stale state to persist for a configurable threshold (e.g. 3 consecutive checks) before triggering redeployment.

---

## 13. Verification Checklist

```bash
# After Phase 1
curl http://localhost:8000/health/live          # → 200 {"status": "alive"}
curl http://localhost:8000/health/ready         # → 200 or 503
curl -H "Authorization: Bearer <admin>" \
     http://localhost:8000/v1/admin/health      # → 200 with full breakdown

# After Phase 4
# SSE stream
curl -N -H "Authorization: Bearer <admin>" \
     http://localhost:8000/v1/admin/health/stream

# History
curl -H "Authorization: Bearer <admin>" \
     "http://localhost:8000/v1/admin/health/history?limit=10"

# Lint + type check
ruff check src/api/services/health/ src/api/routes/health.py src/api/schemas/health.py
mypy src/api/services/health/ --strict

# Tests
pytest tests/unit/test_health_*.py tests/integration/test_health_endpoints.py -v
```
