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
| `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` | `15` | `int (1–60)` | Lifetime of access tokens. Short values improve security; clients must refresh more often. |
| `JWT_REFRESH_TOKEN_EXPIRE_DAYS` | `7` | `int (1–30)` | Lifetime of refresh tokens. Stored in the database; revoked on logout. |
| `JWT_ISSUER` | `apex-api` | `str \| None` | Optional `iss` claim embedded in tokens. Validated on decode. Set to `None` to disable. |

> **⚠️ docker-compose.yml gap:** The `JWT_*` environment variables are currently **missing** from
> `docker-compose.yml`. Add them to the `apex-api` service environment section alongside the
> other settings, or the containerised service will use the insecure placeholder default.

---

### ComfyUI Connection

Controls how Apex connects to the ComfyUI backend that executes generation workflows.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `COMFYUI_HOST` | `127.0.0.1` | `str` | Hostname or IP of the ComfyUI server. Change to the remote host or GPU node IP when ComfyUI is not co-located. |
| `COMFYUI_PORT` | `18188` | `int` | TCP port ComfyUI listens on. Must match the `--port` flag passed to ComfyUI. |

> **Computed:** `COMFYUI_BASE_URL` is derived as `http://{COMFYUI_HOST}:{COMFYUI_PORT}` and
> cannot be set directly.

---

### API Server

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `API_HOST` | `0.0.0.0` | `str` | Interface to bind. Use `127.0.0.1` to restrict to localhost when behind a reverse proxy. |
| `API_PORT` | `8000` | `int` | TCP port the Litestar API listens on. |
| `DEBUG` | `false` | `bool` | Enables debug mode: detailed tracebacks and verbose logging. **Never enable in production.** |

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
