# Apex Configuration Reference

Configuration is loaded from environment variables and/or a `.env` file in the project root.
Variable names are **case-insensitive**. Start from the provided template:

```bash
cp .env.example .env
```

---

## ⚠️ Required Settings (no working defaults)

These must be explicitly set before the service will function correctly in any real environment.

| Variable | Why it's required |
|----------|-------------------|
| `JWT_SECRET_KEY` | Default is a hardcoded placeholder string — **unsafe in any environment**. Signs all access and refresh tokens; must be a strong random secret. |
| `DATABASE_URL` | Default points to `localhost` — must match your actual PostgreSQL instance. |
| `R2_ACCOUNT_ID` | No default — R2 storage is disabled if missing; all upload/output endpoints return `503`. |
| `R2_ACCESS_KEY_ID` | No default — required for R2 S3-compatible authentication. |
| `R2_SECRET_ACCESS_KEY` | No default — required for R2 S3-compatible authentication. |
| `XAI_API_KEY` | No default — Grok image/video generation is disabled if missing. |
| `COMFYUI_HOST` | Default `127.0.0.1` only works when ComfyUI runs on the same machine. |

> **`JWT_SECRET_KEY` action required:** The current default is `"CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET_KEY_256_BITS"`.
> The service starts without error but is **critically insecure** — anyone can forge tokens.
> A `@model_validator` that hard-fails startup on weak/default values should be added to `src/core/config.py`.
> Generate a proper key with:
> ```bash
> python -c "import secrets; print(secrets.token_urlsafe(32))"
> ```

---

## Full Settings Reference

### JWT Authentication

Controls access and refresh token signing. All JWTs are signed with HS256 by default.
Refresh tokens are stored in the database (not as JWTs); only access tokens are issued as JWTs.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `JWT_SECRET_KEY` | `"CHANGE_ME_..."` | `str` | **⚠️ Must be changed.** HMAC secret for signing access tokens. Any change immediately invalidates all existing tokens. |
| `JWT_ALGORITHM` | `HS256` | `str` | JWT signing algorithm. `HS256` is standard; `RS256` requires key pair setup. |
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `15` | `int (1–60)` | Lifetime of access tokens. Short values improve security; clients must refresh more often. Still deliberately not raised even though revocation now exists (see the field's docstring in `src/core/config.py`) — 15 minutes remains the containment window for a credential compromised between revocation checks (e.g. Redis down at the moment of theft). |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | `7` | `int (1–30)` | Lifetime of refresh tokens. Stored in the database; revoked on logout. |
| `JWT_ISSUER` | `apex-api` | `str \| None` | Optional `iss` claim embedded in tokens. Validated on decode. Set to `None` to disable. |

> **⚠️ docker-compose.yml gap:** The `JWT_*` environment variables are currently **missing** from
> `docker-compose.yml`. Add them to the `apex-api` service environment section alongside the
> other settings, or the containerised service will use the insecure placeholder default.

> **Access-token revocation (issue #142):** access tokens are stateless JWTs, but `auth_guard` /
> `content_auth_guard` now also reject a token if `REDIS_URL` is configured and `TokenRevocationService`
> (`src/api/services/token_revocation.py`) says so — either its `iat` is at or before the caller's most
> recent bulk "revoke all sessions" event (`logout-all`, password change/reset, deactivation, refresh-token
> reuse detection), or its specific `jti` was denylisted by a single-device `POST /v1/auth/logout`. See
> `REDIS_URL` below for the fail-open behavior when Redis is unset or unavailable.
>
> **NTP is a hard requirement on every API host.** The bulk-revocation epoch is now written from
> Redis's own clock (`TIME`, round 2 F1b) so multiple API replicas can't disagree with each other about
> *when* a revocation happened — but a token's `iat` is still stamped by the minting instance's local
> clock, and the comparison is `iat <= epoch`. An API host with a clock running noticeably behind Redis's
> can mint tokens that outlive a revocation meant to kill them.

---

### Content Authentication Cookie

Controls `apex_content`, the HttpOnly cookie that authorizes `GET /v1/content/*` (media reads)
without a `Bearer` header — separate from and much longer-lived than the JWT access token above.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `CONTENT_COOKIE_TTL_HOURS` | `24` | `int (1–168)` | Lifetime of the content cookie / its underlying `type: "content"` JWT. Deliberately **not** symmetric with `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` — see `Settings.content_cookie_ttl_hours`'s docstring in `src/core/config.py` for why raising this is safe while raising the access token TTL is not. |
| `CONTENT_COOKIE_NAME` | `apex_content` | `str` | Name of the Set-Cookie key. |

The `Secure` attribute is not independently configurable — it's derived from `debug` (`Settings.content_cookie_secure`), off only in local dev so the cookie is still stored over plain `http://localhost`.

---

### ComfyUI Connection

Controls how Apex connects to the ComfyUI backend that executes generation workflows.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `COMFYUI_HOST` | `127.0.0.1` | `str` | Hostname or IP of the ComfyUI server. Change to the remote host or GPU node IP when ComfyUI is not co-located. |
| `COMFYUI_PORT` | `8188` | `int` | TCP port ComfyUI listens on. Must match the `--port` flag passed to ComfyUI. |

> **Computed:** `COMFYUI_BASE_URL` is derived as `http://{COMFYUI_HOST}:{COMFYUI_PORT}` and
> cannot be set directly.

---

### API Server

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `API_HOST` | `0.0.0.0` | `str` | Interface to bind. Use `127.0.0.1` to restrict to localhost when behind a reverse proxy. |
| `API_PORT` | `8000` | `int` | TCP port the Litestar API listens on. |
| `DEBUG` | `false` | `bool` | Enables debug mode: detailed tracebacks and verbose logging. **Never enable in production.** |

> **⚠️ Same-origin reverse-proxy hazard (`Clear-Site-Data`):** every session-ending endpoint
> (`POST /v1/auth/logout`, `POST /v1/users/me/logout-all`, `POST /v1/users/me/password`,
> `POST /v1/auth/reset-password`, `DELETE /v1/users/me`) sends
> `Clear-Site-Data: "cache", "storage"` (`CLEAR_SITE_DATA_HEADER`,
> `src/api/security/response_headers.py`). `"storage"` clears storage for the origin that
> *sent* the header. Today the API and the frontend are separate origins (`APP_URL` on its own
> host/port vs. `API_HOST`/`API_PORT`, e.g. `api.` vs. the app host in production), so this only
> clears the API origin — which holds nothing but cached content-proxy responses. If the API is
> ever reverse-proxied same-origin with the frontend (e.g. under `/api` on the app domain behind
> the `API_HOST=127.0.0.1` setup above), `"storage"` would instead clear the **frontend's**
> storage on every logout — including service worker registrations, unregistering the PWA worker
> and breaking offline behavior and push until the next load re-registers it. Keep the API on its
> own origin, or drop `"storage"` from the header before ever proxying same-origin.

---

### Database (PostgreSQL)

Apex uses SQLAlchemy 2.0 async via `asyncpg`. The URL **must** use the `postgresql+asyncpg://`
scheme — the synchronous `postgresql://` scheme will fail at startup.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://apex:apex@localhost:5432/apex` | `str` | Full async DSN. Format: `postgresql+asyncpg://<user>:<password>@<host>:<port>/<dbname>` |
| `DB_POOL_SIZE` | `5` | `int` | Persistent connections in the pool. Increase for high request concurrency. |
| `DB_MAX_OVERFLOW` | `10` | `int` | Additional connections allowed under burst load. Total max = `DB_POOL_SIZE + DB_MAX_OVERFLOW`. |
| `DB_ECHO` | `false` | `bool` | Logs every SQL statement via SQLAlchemy. Very verbose — use only to debug queries or migration issues. |

> Run `alembic upgrade head` against the configured database before first start.
> Alembic reads `DATABASE_URL` from settings at runtime.

---

### Cloudflare R2 Storage

User uploads and generation outputs are stored in Cloudflare R2. All three credential
variables must be non-empty or the service starts without storage and upload/output endpoints
return `503`.

Obtain credentials from **Cloudflare Dashboard → R2 → Manage R2 API Tokens**
(Object Read & Write permissions on the target bucket).

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `R2_ACCOUNT_ID` | *(empty)* | `str` | Cloudflare account ID. Found in the dashboard URL or R2 overview page. |
| `R2_ACCESS_KEY_ID` | *(empty)* | `str` | R2 API token access key ID. |
| `R2_SECRET_ACCESS_KEY` | *(empty)* | `str` | R2 API token secret. Only shown once at creation — store it immediately. |
| `R2_BUCKET_NAME` | `apex-user-content` | `str` | Target R2 bucket. Must exist before starting the service — Apex does not create it. |
| `R2_PUBLIC_URL_BASE` | `None` | `str \| None` | Optional custom domain for public file URLs, e.g. `https://cdn.yourdomain.com`. When set, public URLs use this base instead of the default `*.r2.dev` URL. Leave unset when using presigned URLs only. |

> **Storage key layout:**
> - Uploads: `users/{user_id}/uploads/{file_id}.{ext}`
> - Outputs: `users/{user_id}/outputs/{job_id}/{file_id}.{ext}`

---

### Cloudflare R2 Public Assets Bucket

A second, **dedicated public** R2 bucket for static assets served directly to browsers —
currently: cached payment currency logos, fetched from the provider at sync time and
re-served from our own origin so a browser never loads an unvetted third-party image.
This must be a bucket separate from `R2_BUCKET_NAME` above: that bucket is private and
served only through the cookie-authenticated content proxy, and making it public would
gut that auth model entirely.

Shares the same R2 account credentials (`R2_ACCOUNT_ID`/`R2_ACCESS_KEY_ID`/
`R2_SECRET_ACCESS_KEY`) as the user-content bucket — only the bucket name and public
domain differ.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `R2_PUBLIC_ASSETS_BUCKET` | `None` | `str \| None` | Target bucket for public assets. Must exist before starting the service. |
| `R2_PUBLIC_ASSETS_URL_BASE` | `None` | `str \| None` | Public custom domain fronting the bucket, e.g. `https://assets.yourdomain.com`. |

**One-time setup per environment:**
1. Create a new R2 bucket (e.g. `apex-public-assets`), distinct from the user-content bucket.
2. In the Cloudflare dashboard, attach a public custom domain to that bucket.
3. Set both variables above to the bucket name and domain.

Both variables are optional and must be set together — when either is unset, the service
logs a single `payment_currency.logo_cache_disabled` warning at startup and skips logo
caching entirely; currency availability/metadata sync is unaffected, and
`GET /v1/billing/currencies` serves `logo_url: null` for every entry. Cached logos are
uploaded with `Cache-Control: public, max-age=31536000, immutable` (they're content-addressed
by SHA-256, so the same key is always the same bytes).

**R2 API token scope:** the app's R2 API token must carry **Object Read & Write on the
assets bucket in addition to the content bucket** (`R2_BUCKET_NAME`) — a token scoped to
only the content bucket returns `403 Forbidden` (not `404`) on `HeadObject` against the
assets bucket, which is what a scope gap looks like in logs. A misconfigured/unreachable
assets bucket degrades gracefully (per-run circuit breaker after the first storage
failure — see `payment_currency.logo_storage_unavailable` at error level), but startup
also runs a best-effort probe against the assets bucket and logs
`payment_currency.logo_storage_probe_failed` at error level if it can't reach it, so a
scope gap is visible in the first screen of boot logs rather than 33 minutes later inside
a request.

---

### Rate Limiting

Controls rate limits for authentication endpoints using Redis sliding-window counters (or fallback to memory).

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `REDIS_URL` | `None` | `str \| None` | Redis URL (e.g., `redis://localhost:6379/0`). If missing, falls back to single-process in-memory limits. Also backs `TokenRevocationService` (issue #142, access-token revocation — see the JWT Authentication section above): without it, `logout`/`logout-all`/password-change/deactivation/reset still revoke refresh tokens as before, but live access tokens and content cookies are not denylisted until they expire naturally. A Redis outage after startup degrades the same way (fail-open) rather than rejecting authenticated requests — a circuit breaker (see `REDIS_SOCKET_*_TIMEOUT_SECONDS` below) bounds the cost: round 3 trips it in ~0ms on a refused/reset connection (2 consecutive failures) or ~150ms on a slow/hung one (3 consecutive failures, since that case is ambiguous — a GC pause can look the same), reserves exactly one half-open probe per 5s window instead of a thundering herd of concurrent probes, and — as of round 3 — is shared with `get_current_epoch` (the refresh-rotation backstop check), not just `is_revoked`. Logged once as `authrev.circuit_opened` / `authrev.circuit_closed` rather than per request. This checker never gates `GET /health/ready` (round 3) — it is diagnostic-only by design (see `GET /v1/admin/health` for its live status), since failing readiness on it would pull the whole API fleet out of rotation, exactly the outage the fail-open posture exists to prevent. |
| `REDIS_SOCKET_CONNECT_TIMEOUT_SECONDS` | `0.25` | `float (0–5.0]` | TCP connect timeout for the Redis pool (issue #142 F4). `TokenRevocationService.is_revoked`/`get_current_epoch` run on every authenticated request/refresh, so this must stay small — see `src/core/redis.py`'s `init_redis_pool`. |
| `REDIS_SOCKET_TIMEOUT_SECONDS` | `0.05` | `float (0–5.0]` | Redis socket read/write timeout, same rationale (lowered from `0.25` in round 3 — a healthy same-network Redis answers in low single-digit ms, so 50ms is roughly an order of magnitude of headroom; measure your actual `MGET` p99 before raising it). Combined with `retry_on_timeout=False`, a timeout fails straight into `TokenRevocationService`'s fail-open branch instead of silently retrying. |
| `REDIS_HEALTH_CHECK_INTERVAL_SECONDS` | `30.0` | `float (0–300.0]` | How often redis-py proactively PINGs idle pooled connections (issue #142 G3b), so a connection gone stale while unused is caught and replaced before it can produce a spurious `TokenRevocationService` failure mid-request. |
| `RATE_LIMIT_REGISTER` | `5/hour` | `str` | Limit on register endpoint per IP. |
| `RATE_LIMIT_LOGIN` | `10/minute` | `str` | Limit on login endpoint per IP. |
| `RATE_LIMIT_FORGOT_PASSWORD` | `3/hour` | `str` | Limit on forgot-password endpoint per IP. |
| `RATE_LIMIT_RESEND_VERIFICATION` | `3/hour` | `str` | Limit on resend-verification endpoint per IP. |

> **Redis durability (issue #142 F3):** `docker-compose.yml`'s `redis` service runs with
> `--appendonly yes --appendfsync everysec` — without AOF, a Redis restart can lose every revocation
> event (logout-all, password change/reset, reuse detection) written since the last RDB snapshot,
> silently un-revoking credentials that should stay dead. `everysec` bounds worst-case loss to one
> second at a small constant fsync cost — the right trade for a security control. Do not remove this
> flag when customizing Redis deployment outside Docker Compose.

---

### xAI Grok API

Used for AI image and video generation via the xAI SDK (gRPC). When `XAI_API_KEY` is empty
the `GrokClient` raises at construction and all Grok-backed endpoints are unavailable.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `XAI_API_KEY` | *(empty)* | `str` | xAI API key. Required for all Grok image/video generation. Obtain from [console.x.ai](https://console.x.ai). |
| `XAI_API_BASE_URL` | `https://api.x.ai` | `str` | xAI API base URL. Override only if using a proxy or a different region endpoint. |
| `XAI_TIMEOUT` | `300` | `int (30–900)` | gRPC call timeout in seconds. Video generation can be slow; the default 5 min is generous but safe. |
| `GROK_VIDEO_POLL_INTERVAL` | `5` | `int (1–60)` | Seconds between polls when the Grok video worker checks async job status. Lower reduces latency; higher reduces API call volume. |
| `GROK_VIDEO_MAX_POLL_TIME` | `600` | `int (60–1800)` | Maximum total seconds the video worker polls before marking a job as timed out. |
| `GROK_VIDEO_FINALIZATION_LEASE_SECONDS` | `120` | `int (30–900)` | PostgreSQL-time takeover lease for completed-video materialization. Expiry permits a new claimant, but does not invalidate the current token by itself. |
| `GROK_MODERATION_BILLING_POLICY` | `charge` | `charge` \| `refund` | Whether an accepted xAI moderation rejection retains the debit or receives a refund. This is applied by both in-process and standalone video workers. |

> **Materialization migration and operations:** migration 033 adds product-scoped durable
> materialization-attempt and storage-cleanup records. Before it remaps duplicate generation outputs,
> it validates that metadata and tags belong to the canonical output's product and user; ambiguous
> metadata conflicts abort the migration before persistent changes. Tag memberships are merged
> losslessly. The retention worker must continue to run for every configured product so its scoped
> outbox records are reconciled.

---

### Content & Upload Limits

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `RETENTION_DAYS` | `7` | `1–90` | Days before user content is eligible for automatic cleanup. Applied to both R2 objects and database records. |
| `MAX_UPLOAD_SIZE_MB` | `20` | `1–100` | Hard limit on individual upload size in megabytes. Requests exceeding this are rejected with `413` before reading the full body. |

---

### Generation Defaults

Fallback values used when a generation request omits optional parameters. All can be
overridden per-request via the API.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `DEFAULT_STEPS` | `12` | `int` | Denoising steps for the ComfyUI sampler. More steps = slower but potentially higher quality. |
| `MAX_STEPS` | `20` | `int` | Server-side cap on steps a user may request. Prevents runaway generation times. |
| `DEFAULT_CFG` | `1.1` | `float` | Classifier-Free Guidance scale. Higher follows prompt more strictly but can over-saturate. Typical range `1.0–7.0` depending on model. |
| `DEFAULT_SAMPLER` | `euler` | `str` | ComfyUI sampler name. Common values: `euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_sde`. Must match a sampler available in the deployed bundle. |
| `DEFAULT_SCHEDULER` | `beta` | `str` | Noise schedule. Common values: `beta`, `karras`, `normal`, `simple`. Must be compatible with the chosen sampler. |

---

## Environment-Specific Cheatsheets

### Local development (minimal `.env`)

```bash
# Only what differs from defaults
DATABASE_URL=postgresql+asyncpg://apex:apex@localhost:5432/apex
DEBUG=true
DB_ECHO=true

# Generate once and keep in your local .env — never commit this
JWT_SECRET_KEY=<output of: python -c "import secrets; print(secrets.token_urlsafe(32))">

# Omit R2_* and XAI_API_KEY — those subsystems will be disabled but the rest works
```

### Production (minimum required set)

```bash
# Auth — mandatory, no safe default
JWT_SECRET_KEY=<64-char random string>
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=15
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
JWT_ISSUER=apex-api

# Database
DATABASE_URL=postgresql+asyncpg://apex:strongpassword@db-host:5432/apex

# Storage
R2_ACCOUNT_ID=abc123...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=apex-user-content
R2_PUBLIC_URL_BASE=https://cdn.yourdomain.com   # optional

# AI backends
COMFYUI_HOST=<gpu-node-ip>
XAI_API_KEY=<xai-key>                           # optional if Grok not used

# Safety
DEBUG=false
DB_ECHO=false
```
