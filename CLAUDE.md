# CLAUDE.md

## Project Overview

This is the **aisha/apex** ecosystem — an AI content generation service that orchestrates ComfyUI workflows on Vast.ai GPU nodes for image and video generation.

One project lives in this repo:

| Project | Purpose | Entry point | Python |
|---------|---------|-------------|--------|
| **Apex** | Litestar REST API — accepts generation requests, manages jobs, stores content in R2 | `src/api/app.py` → `src/main.py` | 3.13 |

**Aisha** lives in its own separate repo (`gearbox/apex`).

### Multi-Product Architecture

The backend serves two distinct products from the same codebase:

| Product | Slug | Domains | Audience | Content |
|---------|------|---------|----------|---------|
| **vex** | `vex` | `example.com` (+ all subdomains) | Consumer / creator | Permissive — NSFW-capable |
| **Synthara** | `synthara` | `synthara.app` (+ all subdomains) | Enterprise / business | SFW only |

**Key rules:**
- Product is resolved per-request from `Origin` → `Host` → `X-Product-Id` header → localhost fallback
- Domain matching uses eTLD+1 (via `tldextract`): `staging.example.com`, `preview.example.com`, etc. all resolve to `vex` without any config change
- User accounts are **product-scoped**: same email = separate accounts per product
- JWT tokens embed `product_id` claim — tokens are rejected if product doesn't match the request
- All DB tables have a `product_id` column — queries must filter by it
- Product configuration lives in `src/core/product_registry.py` (frozen dataclasses, NOT in DB)

**Adding a new feature:** Every new route handler must inject `product_config: ProductConfig` and/or `product_id: str`, and pass them through to service/repository calls. Every new DB table must include a `product_id VARCHAR(32) NOT NULL` column.

---

## Tech Stack

- **Python 3.13** (Apex — our server, fresh runtime for API + DB)
- **Litestar 2.5+** — async web framework (NOT FastAPI)
- **msgspec** — high-performance serialization for API schemas (NOT Pydantic for DTOs)
- **Pydantic + pydantic-settings** — used ONLY for `Settings` / config loading
- **SQLAlchemy 2.0+** — async ORM with `asyncpg` driver
- **PostgreSQL 16** — database (job queue with `FOR UPDATE SKIP LOCKED`)
- **Alembic** — async migrations
- **Cloudflare R2** — object storage via `aioboto3` (S3-compatible)
- **httpx** — async HTTP client (for ComfyUI communication)
- **uv** — package manager
- **Docker Compose** — local dev and production environments
- **Typer + Rich** — CLI framework (aisha)

---

## Project Structure

```
├── src/
│   ├── api/
│   │   ├── app.py              # Litestar app factory (create_app)
│   │   ├── dependencies/       # DI providers
│   │   │   ├── common.py       # Singleton lifecycle + service container
│   │   │   └── auth.py         # Auth dependencies (user_id, admin_user)
│   │   ├── middleware/
│   │   │   └── product.py      # ProductMiddleware — resolves product per-request
│   │   ├── routes/             # Litestar Controllers
│   │   │   ├── health.py       # HealthController: GET /health/live, /health/ready
│   │   │   │                   # AdminHealthController: GET /v1/admin/health/
│   │   │   │                   #   GET /v1/admin/health/stream (SSE), GET /v1/admin/health/history
│   │   │   ├── content.py      # ContentProxyController: GET /v1/content/outputs/{id}, /uploads/{id}
│   │   │   ├── storage.py      # StorageController: upload, single-item access/download, stats
│   │   │   ├── library.py      # LibraryController: GET /v1/library/, /assets/{asset_ref}[/lineage],
│   │   │   │                   #   /groups/{job_id}; PATCH/PUT/DELETE favorite/delete; POST assets/bulk
│   │   │   ├── library_project.py  # LibraryProjectController: CRUD /v1/library/projects
│   │   │   ├── library_tag.py      # LibraryTagController: CRUD /v1/library/tags
│   │   │   └── push.py         # PushController: GET /v1/push/vapid-public-key
│   │   │                       #   POST/DELETE /v1/push/subscriptions
│   │   ├── schemas/            # msgspec.Struct request/response DTOs
│   │   │   ├── library.py      # LibraryAssetItem/Detail/Patch, LibraryGroupDetail, LibraryLineageGraph,
│   │   │   │                   #   LibraryProject*, LibraryTag*, BulkOperation (tagged union)
│   │   │   └── push.py         # PushSubscriptionRequest/Response, PushNotificationPayload (wire contract)
│   │   ├── security/           # Guards, JWT, password hashing
│   │   │   ├── guards.py       # auth_guard, optional_auth_guard (validates product_id JWT claim)
│   │   │   ├── jwt.py          # JWTService, token encode/decode (embeds product_id claim)
│   │   │   └── password.py     # PasswordService (argon2id)
│   │   └── services/           # Business logic layer
│   │       ├── age_verification.py  # Age gate enforcement (CHECKBOX, DATE_OF_BIRTH)
│   │       ├── comfyui_client.py
│   │       ├── content_proxy.py     # ContentProxyService — ownership check + R2 streaming; delete_content()
│   │       ├── content_retention.py # ContentRetentionService — sweeps expired outputs/uploads (DB-first, R2 best-effort);
│   │       │                        #   also purges orphaned library_asset_metadata/library_asset_tags rows
│   │       ├── library.py           # LibraryService — unified grid/detail, favorites, patch, bulk ops, lineage graph
│   │       ├── library_capabilities.py  # resolve_library_actions() — pure, table-driven available_actions resolver
│   │       ├── library_project.py   # LibraryProjectService — project CRUD (one project per asset)
│   │       ├── library_tag.py       # LibraryTagService — tag CRUD (many-to-many asset tagging)
│   │       ├── health/         # Health check system (Phase 4)
│   │       │   ├── base.py     # HealthChecker protocol, ComponentHealth dataclass
│   │       │   ├── registry.py # HealthCheckRegistry — concurrent execution with timeout
│   │       │   ├── service.py  # HealthService — readiness() / detailed() / check_all_and_build() / persist_snapshot()
│   │       │   ├── worker.py   # HealthSnapshotWorker — periodic checks → DB + Redis publish; leader lock (TTL = max(2x interval, 90s))
│   │       │   ├── stream.py   # health_sse_generator — SSE event generator (Redis Pub/Sub or polling fallback)
│   │       │   ├── history.py  # parse_history_params — history query parsing + ISO 8601 datetime validation (400 on bad input)
│   │       │   └── checkers/
│   │       │       ├── infrastructure.py  # PostgresChecker, RedisChecker, R2Checker
│   │       │       ├── cloud_provider.py  # CloudProviderChecker (ABC), GrokChecker
│   │       │       ├── platform_api.py    # VastAIChecker
│   │       │       └── gpu_session.py     # GpuSessionReconciler — probes active sessions, marks stale
│   │       ├── job_manager.py
│   │       ├── workflow_service.py
│   │       ├── user_content.py
│   │       ├── billing.py      # BillingService (token ledger, account preference)
│   │       ├── pricing.py      # PricingService (cost catalog)
│   │       ├── payment.py      # PaymentService (per-product Stripe/NowPayments keys)
│   │       ├── organization.py # OrganizationService (teams; requires "organizations" feature flag)
│   │       ├── moderation.py   # Provider-specific moderation detectors
│   │       ├── push.py         # PushService — upsert/delete/send; WebPushSender protocol + PywebpushSender
│   │       ├── push_mapping.py # map_event_to_notification() — pure EventEnvelope -> PushNotificationPayload | None
│   │       └── storage/        # R2 storage (r2.py, schemas.py); r2.py has stream_object() context manager
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings); per-product Stripe/NowPayments keys
│   │   ├── enums.py            # ModelType, JobStatus, AspectRatio, Product, LibrarySort, LibraryBadge, etc.
│   │   ├── product.py          # ProductConfig frozen dataclass + policy enums
│   │   ├── product_registry.py # VEX_CONFIG, SYNTHARA_CONFIG, resolve_product_by_domain()
│   │   ├── library_ref.py      # AssetRef, LibraryAssetSource, parse_asset_ref()/format_asset_ref() —
│   │   │                       #   pure typed identity for "upload:<uuid>" / "output:<uuid>"
│   │   └── library_limits.py   # MAX_TAGS_PER_ASSET — shared bound (schemas + services)
│   ├── db/
│   │   ├── models/
│   │   │   ├── storage.py      # GenerationJob (+ lineage FKs), UserImage, GenerationOutput
│   │   │   ├── user.py         # User, RefreshToken
│   │   │   ├── billing.py      # TokenAccount, TokenTransaction, PricingRule, Payment
│   │   │   ├── gpu_session.py  # GpuSession — Vast.ai node lifecycle tracking
│   │   │   ├── health.py       # HealthSnapshot — persisted health check results
│   │   │   ├── push_subscription.py  # PushSubscription — Web Push endpoint + keys
│   │   │   └── library.py      # LibraryAssetMetadata (favorite/title/project_id, polymorphic asset_type+asset_id),
│   │   │                       #   LibraryProject, LibraryTag, LibraryAssetTag (many-to-many join)
│   │   ├── repositories/
│   │   │   ├── user.py         # UserRepository
│   │   │   ├── billing.py      # BillingRepository
│   │   │   ├── job.py          # JobRepository — GenerationJob CRUD
│   │   │   ├── output.py       # OutputRepository — GenerationOutput CRUD
│   │   │   ├── user_image.py   # UserImageRepository — UserImage CRUD + storage stats
│   │   │   ├── health.py       # HealthSnapshotRepository — insert, list_range, cleanup
│   │   │   ├── push_subscription.py  # PushSubscriptionRepository — upsert/delete/list, keyset batch for broadcast
│   │   │   ├── library.py      # LibraryRepository — UNION list_assets over user_images+generation_outputs,
│   │   │   │                   #   lineage walks, metadata upsert/purge (both branches product_id-scoped by construction)
│   │   │   ├── library_project.py  # LibraryProjectRepository
│   │   │   └── library_tag.py      # LibraryTagRepository — CRUD + batched tag lookups + bulk add/remove
│   │   └── session.py          # DatabaseManager, async session factory
│   └── main.py                 # Granian entrypoint
├── config/
│   └── bundles/                # ComfyUI workflow bundles (YYMMDD-nn versioned)
├── alembic/
│   ├── env.py                  # Async migration env
│   ├── script.py.mako          # Migration template
│   └── versions/               # Migration files (sequential: 001, 002, 003, ...)
├── tests/
├── docker-compose.yml          # Base compose config
├── docker-compose.dev.yml      # Dev overrides (hot reload, exposed ports)
├── docker-compose.prod.yml     # Production overrides
├── Dockerfile                  # Multi-stage production build
├── Dockerfile.dev              # Dev build with hot reload
├── Makefile                    # Common commands
├── alembic.ini
├── pyproject.toml              # Apex dependencies
└── .env.example                # Environment template
```

---

## Architecture Principles

### Layered Architecture

```
Routes (Controllers) → Services (Business Logic) → Repository (Data Access) → Models (DB)
         ↓                     ↓                          ↓
   msgspec Structs       Domain logic              SQLAlchemy + asyncpg
```

- **Routes** (`src/api/routes/`): Litestar `Controller` classes. Handle HTTP, validation, response formatting. No business logic.
  - ⚠️ Never name a Litestar handler argument `query`/`headers`/`cookies`/`state`/`scope`/`socket`/`body` — these are reserved kwargs; alias with `Parameter(query=...)` if the wire name must match.
- **Services** (`src/api/services/`): Business logic. Injected via Litestar DI. Services receive repositories/other services as constructor args.
- **Repository** (`src/db/repositories/`): `UserRepository`, `BillingRepository`, `JobRepository`, `OutputRepository`, `UserImageRepository` — all DB queries go here. Receive `AsyncSession`. Never imported directly by routes.
- **Models** (`src/db/models/`): SQLAlchemy 2.0 declarative models using `Mapped[]` and `mapped_column()`.

### Dependency Injection

Litestar DI is configured in `src/api/dependencies/common.py`:

```python
# Singleton services initialized at startup in init_services()
dependencies = {
    # Authentication services
    "auth_service": Provide(get_auth_service, sync_to_thread=False),
    "user_service": Provide(get_user_service, sync_to_thread=False),
    # Core services
    "workflow_service": Provide(get_workflow_service, sync_to_thread=False),
    "settings": Provide(provide_settings, sync_to_thread=False),
    # Storage services
    "r2_storage": Provide(get_r2_storage, sync_to_thread=False),
    "session": Provide(get_db_session),  # request-scoped
    "user_content": Provide(get_user_content, sync_to_thread=False),  # request-scoped
    # Grok services
    "grok_job_service": Provide(get_grok_job_service, sync_to_thread=False),
    # Billing services
    "billing_service": Provide(get_billing_service, sync_to_thread=False),
    "pricing_service": Provide(get_pricing_service, sync_to_thread=False),
    "organization_service": Provide(get_organization_service, sync_to_thread=False),
    "payment_service": Provide(get_payment_service, sync_to_thread=False),
    # Email services
    "email_verification_service": Provide(get_email_verification_service, sync_to_thread=False),
    # Unified job / generation services
    "unified_job_service": Provide(get_unified_job_service, sync_to_thread=False),
    "generation_service": Provide(get_generation_service, sync_to_thread=False),
    # Real-time events
    "event_bus": Provide(get_event_bus, sync_to_thread=False),
    "sse_ticket_service": Provide(get_sse_ticket_service, sync_to_thread=False),
    # Product context (set by ProductMiddleware, extracted here as DI)
    "product_config": Provide(get_product_config, sync_to_thread=False),  # ProductConfig
    "product_id": Provide(get_product_id, sync_to_thread=False),  # str slug
    # JWT service (needed by auth routes to mint content tokens)
    "jwt_service": Provide(get_jwt_service, sync_to_thread=False),
    # Content proxy + library (request-scoped)
    "content_proxy": Provide(get_content_proxy, sync_to_thread=False),
    "library_service": Provide(get_library_service, sync_to_thread=False),
    "library_project_service": Provide(get_library_project_service, sync_to_thread=False),
    "library_tag_service": Provide(get_library_tag_service, sync_to_thread=False),
    # Idempotency
    "idempotency_service": Provide(get_idempotency_service, sync_to_thread=False),
    # Health
    "health_service": Provide(get_health_service, sync_to_thread=False),
    # GPU sessions
    "gpu_session_service": Provide(get_gpu_session_service, sync_to_thread=False),
    # Internal provisioning callbacks (node bearer auth validated in handler)
    "provisioning_callback_service": Provide(
        get_provisioning_callback_service, sync_to_thread=False
    ),
    # Web Push (503 via get_push_service when VAPID keys/Redis not configured)
    "push_service": Provide(get_push_service, sync_to_thread=False),
}
```

Note: `JobRepository`/`OutputRepository`/`UserImageRepository` are constructed directly inside the services that need them (not DI-provided) — only the `AsyncSession` itself is injected.

`BillingService` is a single process-wide singleton (`ServiceContainer.billing_service`, built once in `init_services()`): it is stateless and side-effect-free w.r.t. real-time events (a pure event-builder — see the Billing section), so nothing prevents sharing one instance across all callers.

Controller-scoped dependencies (in `src/api/dependencies/auth.py`):
- `get_current_user_id` — extracts UUID from `request.state` (set by `auth_guard`)
- `get_current_admin_user` — loads User via `UserRepository`, verifies `is_admin` flag

Lifecycle: `lifespan` context manager in `app.py` calls `init_services()` / `shutdown_services()`.

### Key Conventions

1. **msgspec for API schemas, Pydantic only for Settings**
   - All request/response DTOs are `msgspec.Struct` with `kw_only=True`
   - Use `msgspec.Meta` for validation constraints: `Annotated[int, msgspec.Meta(ge=1, le=20)]`
   - Use `forbid_unknown_fields=True` on request structs
   - Pydantic is exclusively for `Settings(BaseSettings)` in `src/core/config.py`

2. **Enums in `src/core/enums.py`** — all `str, Enum` for JSON compatibility:
   - `ModelType`, `GenerationType` (t2i, i2i, t2v, i2v, flf2v, v2v), `JobStatus`, `AspectRatio`
   - `TransactionType` (debit, credit, refund, admin_adjust), `AccountType`, `PaymentStatus`, `SubscriptionTier`
   - `Product` (vex, synthara)
   - `OutputMediaType` (image, video), `LibraryBadge` (image, prompt), `LibraryGroupSourceType` (upload, output)
   - `LibrarySort` (newest, oldest, expiring_soon) — `LibraryAssetSource` (upload, output) lives in `src/core/library_ref.py`, not `enums.py`, since it's paired with the pure `AssetRef`/`parse_asset_ref` identity helpers there

3. **Async everywhere** — all DB, HTTP, and storage operations are async.

4. **SQLAlchemy 2.0 style** — `Mapped[]` type annotations, `mapped_column()`, async sessions via `asyncpg`.

5. **Repository pattern** — `BaseRepository` provides shared `get`-with-optional-owner and cursor pagination helpers. Domain repositories (`JobRepository`, `OutputRepository`, `UserImageRepository`, `UserRepository`, `BillingRepository`, `LibraryRepository`, `LibraryProjectRepository`, `LibraryTagRepository`) wrap all DB queries. Injected with `AsyncSession`. All query methods must accept and filter by `product_id` where applicable.

6. **Authentication pattern** — `auth_guard` validates JWT and sets `user_id` in connection state. Admin authorization uses a DI dependency (`get_current_admin_user`) instead of a guard, so it uses the request-scoped session and repository layer.

7. **R2 storage keys** follow the pattern:
   - Uploads: `users/{user_id}/uploads/{file_id}.{ext}`
   - Outputs: `users/{user_id}/outputs/{job_id}/{file_id}.{ext}`

8. **Idempotency pattern** — three mutation endpoints require `Idempotency-Key` header:
   - `POST /v1/generate/`, `POST /v1/billing/topup/stripe`, `POST /v1/billing/topup/nowpayments`, `POST /v1/admin/accounts/{id}/adjust`
   - Implementation: `src/api/services/idempotency.py` (service), `src/db/repositories/idempotency.py` (repo), `src/db/models/idempotency.py` (model)
   - Pattern: call `idempotency_service.check()` first → if `IdempotencyReplayResult` return cached → else proceed → call `complete()` after commit, `fail()` on exception
   - Keys are scoped to `(user_id, product_id)`, TTL 24h (configurable via `IDEMPOTENCY_KEY_TTL_HOURS`)
   - Top-up endpoints also enforce a per-product payment-provider guard (`ProductConfig.supports_payment_provider()`) **before** `idempotency_service.check()`, so a 403-rejected request never consumes the idempotency key.

9. **Billing events are post-commit only** — `BillingService` (`src/api/services/billing.py`) is a stateless, side-effect-free event-builder: every ledger-mutating method (`check_and_reserve`, `credit`, `refund`, `partial_refund`, `admin_adjust`, `settle_session_usage`) returns a `BalanceEvent` (via `BillingResult`/`SettleUsageResult`) instead of publishing it. Callers **must** publish it via `EventBus.publish_balance(event)` strictly *after* `session.commit()` succeeds — publishing before commit would let a rolled-back transaction produce a phantom balance update to SSE subscribers. `EventBus.publish_balance` no-ops safely on `None` and never raises.
   - NowPayments IPN underpayments/overpayments auto-credit proportionally via a ledger-delta (never held for manual review) — `PaymentStatus.PARTIALLY_PAID` is informative and non-terminal; only `COMPLETED` blocks IPN reprocessing.
   - Balance-event targets are resolved from the account row (owner / org members), not from the caller. `credit()` takes no `user_id` param — like `admin_adjust`, it resolves SSE targets via the shared `BillingService._resolve_event_targets()` helper (personal account → `account.user_id`; enterprise account → all org members).

10. **Product-awareness in new code** — when adding new routes/services:
   - Inject `product_config: ProductConfig` and/or `product_id: str` via Litestar DI
   - Pass `product_id` to all repository and service calls
   - Every new DB model must have `product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)`
   - Every new DB table in migrations must include `sa.Column("product_id", sa.String(32), nullable=False)`

11. **Tiered free-amount top-up** — `POST /v1/billing/topup/{stripe,nowpayments}` take `amount_usd: int` (whole USD, not a `package_id`). Discount tiers (`Settings.billing_pricing_tiers`, e.g. `{10: 0, 50: 0, 100: 5, 250: 10}`) apply the highest threshold at or below the amount to the **price paid**, never the token count — `tokens_granted = amount_usd * Settings.billing_tokens_per_usd` is always the full pre-discount value; `Payment.amount_usd` stores `total_due` (the discounted charge). `src/core/topup_pricing.py` (`build_quote`, `resolve_discount`, `topup_tiers_for`) is the single pure-math source of truth — both `PaymentService` (the actual charge) and `GET /v1/billing/topup/options` (the UI summary) call it, so they can never disagree. NowPayments IPN crediting is proportional and **uncapped** on overpayment (`target = floor(tokens_granted * ratio)`, ratio in `[0.99, 1.01]` snaps to full) — deliberately overpaying a low-tier invoice buys tokens at that tier's worse discount than the correct higher-tier invoice would, so uncapped crediting is arbitrage-free. There is no DB-backed tier table or admin CRUD yet (env-config + restart); `topup_tiers_for(product_id, settings)` is the seam for per-product pricing later.

12. **Cursor-only pagination** — all paginated endpoints use keyset (cursor) pagination via `CursorPage[T]` with the `limit+1` fetch pattern. **Never use `OFFSET`-based pagination.** Cursor pagination is stable under concurrent inserts, avoids O(offset) scans, and is the project standard established in `docs/designs/pagination-unification-design.md`.
    - Repository `list_*` methods accept `cursor_ts: datetime | None` and `cursor_id: UUID | None` — not `offset`.
    - `BaseRepository._list_by_user_cursor()` provides the shared implementation.
    - Cursor encoding/decoding lives in `src/api/schemas/pagination.py` (`encode_cursor` / `decode_cursor`).
    - The `has_more` flag is derived from `len(results) > limit` — no `COUNT(*)` queries for pagination.

13. **Library asset identity & capabilities** — `src/core/library_ref.py`'s `AssetRef`/`parse_asset_ref`/`format_asset_ref` are the only way a client-supplied `asset_ref` (`"upload:<uuid>"` / `"output:<uuid>"`) is resolved: parse → typed lookup → ownership + product check, never a bare-UUID try-both-tables lookup. `LibraryRepository.list_assets` is a `UNION ALL` over `user_images` + `generation_outputs`, keyed by a 3-part cursor (`encode_library_cursor`/`decode_library_cursor`, alongside the regular cursor helpers in `src/api/schemas/pagination.py`) so pagination has one strict total order across both tables — a cursor is rejected if replayed under a different `sort`. `available_actions` on every library item is resolved by `resolve_library_actions()` (`src/api/services/library_capabilities.py`) — a pure, table-driven function of media type + whether the asset has generation metadata; never infer actions from `source` in route/service code. `library_asset_metadata`/`library_asset_tags` are polymorphic (`asset_type` + `asset_id`, no FK) — any path that deletes an asset (single delete, bulk delete, or the retention sweeper) must also purge its metadata/tag rows explicitly, or they leak as orphans.

---

## Common Commands

```bash
# Development
make dev                          # Start dev env (docker compose + hot reload)
make down                         # Stop containers
make logs                         # Follow logs
make shell                        # Shell into API container
make db-shell                     # psql into PostgreSQL

# Database
make migrate                      # Run alembic upgrade head
make migrate-new NAME=add_x       # Auto-generate migration
make migrate-down                 # Downgrade one step

# Testing & Quality
make test                         # pytest -v
make test-cov                     # pytest with coverage
make test-integration             # run DB integration tests in a docker container: pytest tests/integration/ -v --tb=short
make test-all                     # run full test suit in a docker container: DB integration + unit + coverage
make lint                         # linters check (ruff, mypy, pyright) + format check
make format                       # ruff format
```

---

## Database Conventions

- **Async migrations** — `alembic/env.py` uses `async_engine_from_config` + `asyncio.run()`
- **Migration naming** — sequential IDs: `001_initial_migration.py`, `002_add_auth_tables.py`
- **UUIDs as primary keys** — `PG_UUID(as_uuid=True)` everywhere
- **Timestamps** — `DateTime(timezone=True)` with `server_default=text("CURRENT_TIMESTAMP")`
- **Composite indexes** — defined in `__table_args__` tuples
- **Soft patterns**: `FOR UPDATE SKIP LOCKED` for job queue assignment (prevents race conditions)

### Models

| Model | Table | Purpose |
|-------|-------|---------|
| `User` | `users` | User accounts with auth, admin flag; `preferred_billing_account` stores personal/enterprise preference |
| `RefreshToken` | `refresh_tokens` | JWT refresh token rotation |
| `GenerationJob` | `generation_jobs` | Job queue: status tracking, prompts, analytics; `source_job_id`, `source_output_id`, `input_image_id` for lineage |
| `UserImage` | `user_images` | Uploaded input images metadata |
| `GenerationOutput` | `generation_outputs` | Generated output images metadata |
| `TokenAccount` | `token_accounts` | Token balance accounts (personal or enterprise) |
| `TokenTransaction` | `token_transactions` | Append-only token ledger (debits, credits, refunds) |
| `PricingRule` | `pricing_rules` | Per-model token cost catalog |
| `Payment` | `payments` | Stripe payment records |
| `Organization` | `organizations` | Teams with shared token accounts |
| `OrganizationMember` | `organization_members` | Org membership + roles (owner/admin/member) |
| `IdempotencyKey` | `idempotency_keys` | Request deduplication; stores cached responses; scoped to `(user_id, product_id, key)` |
| `GpuSession` | `gpu_sessions` | Vast.ai GPU node lifecycle; health reconciler marks unreachable sessions as `stale`; future billing will add cost columns |
| `HealthSnapshot` | `health_snapshots` | Persisted health check results; written by `HealthSnapshotWorker` every `HEALTH_SNAPSHOT_INTERVAL_SECONDS`; no `product_id` (global); queried by `GET /v1/admin/health/history` |
| `PushSubscription` | `push_subscriptions` | Web Push endpoint + keys (`p256dh`/`auth`); `endpoint` unique — upsert reassigns ownership on shared-device/account-switch |
| `LibraryAssetMetadata` | `library_asset_metadata` | Per-asset favorite/display_title/project_id; polymorphic (`asset_type` + `asset_id`, no FK) over `user_images`/`generation_outputs`; rows created lazily on first mutation |
| `LibraryProject` | `library_projects` | User-created grouping; one project per asset (nullable FK on `LibraryAssetMetadata`, `ON DELETE SET NULL`); name unique case-insensitively per owner |
| `LibraryTag` | `library_tags` | User-created tag; name unique case-insensitively per owner |
| `LibraryAssetTag` | `library_asset_tags` | Many-to-many join: one asset tagged with one tag; polymorphic like `LibraryAssetMetadata`; `ON DELETE CASCADE` from `LibraryTag` |

---

## Background Workers

All periodic background loops (token cleanup, Aisha/Grok job polling, GPU provisioning, orphan/billing reconciliation, health snapshots) subclass `PeriodicWorker` (`src/workers/base.py`), which owns the uniform lifecycle: `start()`/`stop()` with graceful drain, jittered sleeps, a heartbeat (`last_tick_at`/`last_tick_duration_ms`), and an optional Redis-backed leader lease (`LeaderLease`, same file) so only one process ticks in multi-worker Granian deployments. Business logic stays in each worker's `run_once()`; constructor DI signatures are unchanged (no service-locator).

- **Leader lease**: owned (per-instance token, `SET NX EX` acquire + compare-and-swap renew/release), TTL = `max(3 × interval, 90s)`, key `{environment}:worker:{name}:lease` (built by `lease_key()`, `src/workers/base.py` — single construction site, namespaced by `Settings.environment` so staging and production can never elect a shared leader). Fails CLOSED on Redis errors: a process that cannot verify its leadership does not act as leader, except within a short post-renewal grace window where its key is provably still live in Redis. Fail-open would let any process with a broken Redis connection promote itself to leader of every worker at once — the failure mode this design prevents. The lease namespacing does not extend to the rest of the Redis keyspace (pub/sub channels, `rate_limit:`, `sse_ticket:`, etc. are still global) — see the `LeaderLease` docstring.
- **Redis deployment invariant**: every environment (development, staging, production) runs its own Redis container; there is no shared Redis today and none is planned. This is why the lease/keyspace namespacing above is defensive rather than load-bearing — see `src/core/redis.py`'s module docstring. Do not consolidate environments onto one Redis instance; keys and pub/sub channels are deliberately unnamespaced outside the lease prefix.
- **Two Redis connection pools** (`src/core/redis.py`): the default pool (`init_redis_pool`/`get_redis_client`, bound by `Settings.redis_max_connections`, default 50) serves only short-lived operations — token revocation, worker leases, SSE ticket lookups. The SSE pool (`init_sse_redis_pool`/`get_sse_redis_client`, bound by `Settings.redis_sse_max_connections`, default 500) is dedicated to `EventBus.subscribe`, which holds one connection per connected SSE client for the life of the stream. They're isolated because redis-py raises `ConnectionError` (a `RedisError`) on pool exhaustion rather than blocking — a shared pool exhausted by SSE load would push `TokenRevocationService.is_revoked` into its fail-open branch (silently accepting revoked JWTs) and `LeaderLease` into its fail-closed branch (stalling every worker). Both pool bounds multiply by `ASGI_WORKERS` per host — check against Redis `maxclients` before raising either. `init_redis_pool`/`init_sse_redis_pool` are idempotent for a repeat call with the same URL and raise on a repeat call with a different URL (re-initializing would silently orphan the previous pool's connections).
- **Rate limiter failure posture**: `RateLimitMiddleware` (`src/api/middleware/rate_limit.py`) fails OPEN on Redis storage errors from `limiter.hit()`/`get_window_stats()` — logs `rate_limit.storage_error` at `warning` and lets the request through with no `X-RateLimit-*` headers, rather than 500ing login/register/content-cookie. This matches `TokenRevocationService.is_revoked`'s established fail-open posture for the same dependency (rate limiting is abuse control, not authorization). The rate limiter uses its own independent Redis client (`limits.aio.storage.RedisStorage`, not the pools above) but is configured with the same `Settings`-derived socket timeouts so it can't stall the auth path waiting on a hung backend.
- **`WORKER_MODE`** (env, `Settings.worker_mode`): `all` (default) starts every worker alongside the API; `api_only` starts none, for a process that only serves HTTP while workers run elsewhere. `asgi_workers > 1` with `WORKER_MODE=all` requires `REDIS_URL` (enforced by a `Settings` startup validator) — without it every Granian process would run every worker with no leader election.
- **Liveness**: `WorkerHeartbeatChecker` (`src/api/services/health/checkers/workers.py`, category `ComponentCategory.workers`) reports a worker stale if it's running, holds the leader lease, and hasn't ticked within `max(3 × interval, 120s)` (with a startup grace period for workers that haven't ticked yet). Non-leader and stopped workers are never reported stale. Registered into the same `HealthCheckRegistry` surfaced at `/v1/admin/health`.
- **Grok video worker**: also runnable standalone via `python -m src.workers.grok_video` — it shares the same lease key as the in-process worker, so running both simultaneously is safe.
- **Push dispatcher** (`src/workers/push_dispatcher.py`): fans out real-time events as Web Push notifications. Deliberately does **not** subclass `PeriodicWorker` — it's a long-lived Redis Pub/Sub read loop (`psubscribe("user:*")` + `subscribe("system:broadcast")`), not a tick — but reuses the same `LeaderLease` so only one process holds the subscription in multi-worker deployments. Starts only when `Settings.push_enabled` (VAPID keys + `REDIS_URL` all set). See §15b of `docs/BACKEND_API_REFERENCE.md` for the event → notification mapping and wire contract.
- **Content retention sweeper** (`ContentRetentionWorker`, `src/workers/content_retention.py` + `ContentRetentionService`, `src/api/services/content_retention.py`): enforces `expires_at` on `generation_outputs`/`user_images` — deletes expired DB rows first (bulk core `DELETE`, batched via `Settings.content_cleanup_batch_size`/`content_cleanup_max_batches_per_run`), soft-deletes jobs left with zero outputs, purges the matching `library_asset_metadata`/`library_asset_tags` rows (polymorphic, no FK — orphaned otherwise), commits, then best-effort deletes the matching R2 objects (`R2StorageService.delete_many`). Deliberately opts out of the leader lease (`use_leader_lease=False`, D5) — sweep deletes are idempotent (a row already gone isn't reselected; an R2 delete of a missing key is a no-op), so duplicate ticks across instances are harmless; revisit if Apex goes multi-replica and duplicate R2 delete traffic becomes worth avoiding. Starts only when `Settings.content_cleanup_enabled` and R2 is configured. ⚠️ With the default `RETENTION_DAYS=7`, all user generations are deleted a week after creation — confirm the production retention value before enabling in prod.

---

## Bundle System (Aisha)

Bundles are versioned, reproducible ComfyUI environment configurations:

```
config/bundles/{bundle_name}/{version}/
├── bundle.yaml          # Metadata, ComfyUI commit, custom nodes, models
├── requirements.lock    # Frozen pip dependencies
├── workflow.json        # ComfyUI workflow (GUI or API format)
└── extra_model_paths.yaml  # Optional model path overrides
```

- **Version format**: `YYMMDD-nn` (e.g., `260101-01`)
- **Current symlink**: `config/bundles/{name}/current -> {version}/`
- **Deploy modes**: full, models-only, dry-run

---

## Environment Variables

Critical env vars (see `.env.example` for full list):

```bash
# Database
DATABASE_URL=postgresql+asyncpg://apex:apex@localhost:5432/apex

# Cloudflare R2
R2_ACCOUNT_ID=...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=apex-user-content
CONTENT_URL_TTL=10800           # Cache-Control max-age for content proxy (seconds, default 3h)
IMAGE_MAX_INPUT_MEGAPIXELS=100.0  # Decompression-bomb gate: max decoded pixel count (MP) for input images (upload + i2i remix), enforced before full decode

# Content retention sweeper (enforces RETENTION_DAYS)
CONTENT_CLEANUP_ENABLED=true             # Enable the periodic content retention sweeper
CONTENT_CLEANUP_INTERVAL_SECONDS=900     # Seconds between sweep runs (60-86400)
CONTENT_CLEANUP_BATCH_SIZE=500           # Max expired rows fetched per batch, per content type (1-5000)
CONTENT_CLEANUP_MAX_BATCHES_PER_RUN=20   # Upper bound on batches drained per run — backlog control (1-200)

# Health snapshot persistence (Phase 4)
HEALTH_SNAPSHOT_INTERVAL_SECONDS=60   # Snapshot cycle interval (10–600s, default 60)
HEALTH_SNAPSHOT_RETENTION_DAYS=30     # Retention period for health_snapshots table

# Web Push (all three + REDIS_URL required to enable — see Settings.push_enabled)
VAPID_PUBLIC_KEY=...           # Generate with: uv run python tools/generate_vapid_keys.py
VAPID_PRIVATE_KEY=...          # Never commit — secrets/env only
VAPID_SUBJECT=mailto:ops@apex.ai
PUSH_BROADCAST_CONCURRENCY=10  # Max concurrent Web Push sends per broadcast fan-out batch

# Aisha-specific
ACS_COMFYUI_PATH=/workspace/ComfyUI
ACS_BUNDLES_PATH=config/bundles
ACS_HF_TOKEN=...            # HuggingFace (private models)
ACS_CIVITAI_API_TOKEN=...   # Civitai model downloads
```

### NowPayments dashboard requirements

Per-product dashboard config that has no code-side representation — must be set manually:

- **IPN format: any of them.** `NowPaymentsGateway.verify_webhook` HMAC-verifies against the raw wire bytes first, then against a lexeme-preserving canonical re-serialization (`src/api/services/payments/ipn_canonical.py`) that reproduces NowPayments' signed bytes regardless of whether numbers arrive as JSON literals ("Classic"), quoted strings ("All-Strings"), or mixed. The dashboard toggle is no longer a correctness dependency — historically this implementation required "All-Strings"; that constraint is gone.
- **IPN callback URL is no longer a dashboard responsibility.** Apex sends an explicit per-invoice `ipn_callback_url` (`Settings.nowpayments_ipn_callback_url`, e.g. `https://api.example.com/v1/billing/webhooks/nowpayments`) on every invoice-creation call, which NowPayments prefers over the store-level dashboard URL. This is required config — invoice creation raises (500) if it's unset — because staging and prod share one NowPayments account and a single dashboard field can't route both. Keep the IPN *secret* configured per-product as before (unrelated to the callback URL). The dashboard's "resend IPN" button remains the recovery tool for missed deliveries and is safe against our idempotent handler.
- **Payout currency + payout wallet address** configured per product — settlement conversion from whatever currency the customer paid in happens on the NowPayments side, not in Apex.
- **Enabled-currencies list** — Apex never validates `pay_currency` against a hardcoded list; NowPayments' own 4xx on invoice creation is the validator.

---

## Code Style & Tooling

- **Line length**: 100
- **Linter**: `ruff` (E, W, F, I, B, C4, UP, ARG, SIM, S, ISC, PIE, RET, ASYNC, DTZ, RUF, LOG, G, TRY, T20, TID, PLE, PLW, FA, PGH, TC, FLY, ANN)
- **Formatter**: `ruff format`
- **Type checker**: `mypy --strict` with SQLAlchemy plugin
- **Test runner**: `pytest` with `asyncio_mode = "auto"`
- **Import order**: `ruff` isort with `known-first-party = ["src"]`
- **Ruff target**: `py313

### Style Rules

- Use `from __future__ import annotations` in modules with forward refs
- Use `TYPE_CHECKING` guard for import-only types
- Docstrings: Google style with Args/Returns/Raises sections
- Prefer `X | None` over `Optional[X]`
- Use `Sequence` from `collections.abc` for read-only collection returns
- Prefer `dataclass` for internal value objects, `msgspec.Struct` for API boundaries
- No `f"..."` in log calls — use `logger.info("message %s", var)` format (lazy evaluation)

---

## Exception Handling & Logging

All exception handlers live in `src/api/app.py` and produce a unified `ErrorEnvelope` response.

### Logging severity by handler type

| Handler | Log level | Rationale |
|---------|-----------|-----------|
| `http_exception_handler` — 4xx | `logger.warning` | Expected client errors, tracked for pattern analysis |
| `http_exception_handler` — 5xx | `logger.error` with `exc_info=` | Server errors, needs investigation |
| Billing / moderation / org handlers | `logger.warning` | Domain errors, expected in normal operation |
| `idempotency_conflict_handler` | `logger.info` | Normal race condition, not an error |
| `global_exception_handler` | `logger.error` with `exc_info=` | Fully unhandled exception, always critical |

### Global catch-all

`global_exception_handler` is registered as `Exception: global_exception_handler` (last in the `exception_handlers` dict — must stay last due to MRO resolution). It:
- Returns a hardcoded `"An unexpected error occurred."` message — never leaks `str(exc)` to the client
- Logs `exc_type`, `path`, `method`, and full traceback via `exc_info=exc`

### Log event naming

Dot-separated, structured fields as kwargs — no f-strings:
- `"http.error"` — HTTP exceptions
- `"billing.insufficient_balance"`, `"billing.account_not_found"`, etc. — billing domain
- `"moderation.rejected"` — moderation errors
- `"payment.verification_failed"` — payment verification
- `"organization.permission_denied"`, `"organization.balance_nonzero"` — org errors
- `"idempotency.conflict"` — idempotency key collision
- `"unhandled_exception"` — global catch-all

### Request body size limit

`Litestar(request_max_body_size=settings.max_upload_size_bytes)` is set in `create_app()`. This uses `Settings.max_upload_size_bytes` (computed from `max_upload_size_mb * 1024 * 1024`, default 20 MB). Payloads exceeding this return `413 Content Too Large` at the ASGI level before any route handler runs.

---

## Testing

```bash
# Run all tests
pytest -v

# Run with coverage
pytest --cov=src --cov-report=term-missing

# Run specific test file
pytest tests/test_schemas.py -v
```

- **Fixtures**: Use `tmp_path` for filesystem tests, factory fixtures for models
- **Async tests**: `pytest-asyncio` with `asyncio_mode = "auto"` (no `@pytest.mark.asyncio` needed)
- **Mocking R2**: Use `moto[s3]` for S3-compatible storage mocks
- **Schema validation**: Test via `msgspec.json.decode()` to exercise constraint validation

---

## Docker

```bash
# Dev (hot reload, exposed DB port)
docker compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Build image
docker build -t apex-api:latest .
```

- Multi-stage Dockerfile: builder (uv + deps) → runtime (slim)
- Non-root `appuser` in production
- Health check on `/health/live` (liveness probe)
- `PYTHONPATH=/app` in container
