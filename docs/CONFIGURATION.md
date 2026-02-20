# Apex Configuration Reference

Configuration is loaded from environment variables and/or a `.env` file in the project root.
Variable names are **case-insensitive**. Start from the provided template:

```bash
cp .env.example .env
```

---

## ⚠️ Required Settings (no working defaults)

These must be set before the service will function correctly. Everything else has a
reasonable default that works for local development.

| Variable | Why it's required |
|----------|-------------------|
| `DATABASE_URL` | Default points to `localhost` — must match your actual PostgreSQL instance in any non-trivial environment |
| `R2_ACCOUNT_ID` | No default — R2 storage is unavailable without it; upload/output endpoints will be disabled |
| `R2_ACCESS_KEY_ID` | No default — required to authenticate with the R2 S3-compatible API |
| `R2_SECRET_ACCESS_KEY` | No default — required to authenticate with the R2 S3-compatible API |
| `SECRET_KEY` ¹ | **Not yet in config** — must be added before shipping auth; used to sign JWT access/refresh tokens |
| `COMFYUI_HOST` | Default `127.0.0.1` works only when ComfyUI runs on the same machine |

> ¹ `SECRET_KEY` is architecturally required (OAuth2 PKCE / JWT) but is not yet present in
> `src/core/config.py`. See the [Missing Settings](#missing-settings) section below.

---

## Full Settings Reference

### ComfyUI Connection

Controls how Apex connects to the ComfyUI backend that executes generation workflows.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `COMFYUI_HOST` | `127.0.0.1` | `str` | Hostname or IP of the ComfyUI server. Change to the remote host when ComfyUI runs on a separate GPU node. |
| `COMFYUI_PORT` | `18188` | `int` | TCP port ComfyUI listens on. Must match the `--port` flag passed to ComfyUI. |

> **Computed:** `COMFYUI_BASE_URL` is derived as `http://{COMFYUI_HOST}:{COMFYUI_PORT}` and is
> not set directly.

---

### API Server

Controls the Litestar HTTP server itself.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `API_HOST` | `0.0.0.0` | `str` | Interface to bind. `0.0.0.0` listens on all interfaces; use `127.0.0.1` to restrict to localhost only (e.g. behind a reverse proxy). |
| `API_PORT` | `8000` | `int` | TCP port the API listens on. |
| `DEBUG` | `false` | `bool` | Enables Litestar debug mode: detailed tracebacks, auto-reload, and verbose logging. **Never enable in production.** |

---

### Database (PostgreSQL)

Apex uses SQLAlchemy 2.0 async with `asyncpg`. The URL **must** use the `postgresql+asyncpg://` scheme — the synchronous `postgresql://` scheme will fail.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://apex:apex@localhost:5432/apex` | `str` | Full async DSN. Format: `postgresql+asyncpg://<user>:<password>@<host>:<port>/<dbname>` |
| `DB_POOL_SIZE` | `5` | `int` | Number of persistent connections in the pool. Increase for high concurrency. |
| `DB_MAX_OVERFLOW` | `10` | `int` | Additional connections allowed beyond `DB_POOL_SIZE` under burst load. Total max connections = `DB_POOL_SIZE + DB_MAX_OVERFLOW`. |
| `DB_ECHO` | `false` | `bool` | Logs every SQL statement to stdout via SQLAlchemy. Very verbose — use only during development to debug queries or migration issues. |

> **Migrations:** Alembic reads `DATABASE_URL` from settings at runtime (`alembic/env.py`).
> Run `alembic upgrade head` after setting this variable before first start.

---

### Cloudflare R2 Storage

Apex stores user uploads and generation outputs in Cloudflare R2. All three credential
variables must be set together — if any are empty the service starts without storage and all
upload/output endpoints return `503`.

Get credentials from **Cloudflare Dashboard → R2 → Manage R2 API Tokens**.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `R2_ACCOUNT_ID` | *(empty)* | `str` | Your Cloudflare account ID. Found in the dashboard URL or the R2 overview page. |
| `R2_ACCESS_KEY_ID` | *(empty)* | `str` | R2 API token access key ID. Created under "Manage R2 API Tokens" with Object Read & Write permissions on the target bucket. |
| `R2_SECRET_ACCESS_KEY` | *(empty)* | `str` | R2 API token secret. Only shown once at creation time — store it immediately. |
| `R2_BUCKET_NAME` | `apex-user-content` | `str` | Name of the R2 bucket that holds user content. Must exist before starting the service; Apex does not create it automatically. |
| `R2_PUBLIC_URL_BASE` | `None` | `str \| None` | Optional custom domain for serving files publicly, e.g. `https://cdn.yourdomain.com`. When set, public URLs use this base instead of the R2 default `*.r2.dev` URL. Requires a custom domain configured in R2. Leave unset if using presigned URLs only. |

> **Storage key structure:**
> - Uploads: `users/{user_id}/uploads/{file_id}.{ext}`
> - Outputs: `users/{user_id}/outputs/{job_id}/{file_id}.{ext}`

---

### Content & Upload Limits

| Variable | Default | Range | Description |
|----------|---------|-------|-------------|
| `RETENTION_DAYS` | `7` | `1–90` | Days before user content (uploads and outputs) is eligible for automatic cleanup. Applies both to R2 objects and database records. |
| `MAX_UPLOAD_SIZE_MB` | `20` | `1–100` | Hard limit on individual file upload size in megabytes. Requests exceeding this are rejected with `413` before reading the body. |

---

### Generation Defaults

Fallback values used when a generation request omits optional parameters. All can be
overridden per-request via the API.

| Variable | Default | Type | Description |
|----------|---------|------|-------------|
| `DEFAULT_STEPS` | `12` | `int` | Denoising steps for the sampler. More steps = slower but potentially higher quality. |
| `MAX_STEPS` | `20` | `int` | Server-side cap on steps a user may request. Prevents runaway generation time. |
| `DEFAULT_CFG` | `1.1` | `float` | Classifier-Free Guidance scale. Higher values follow the prompt more strictly but can over-saturate. Typical range: `1.0–7.0` depending on the model. |
| `DEFAULT_SAMPLER` | `euler` | `str` | ComfyUI sampler name. Common values: `euler`, `euler_ancestral`, `dpmpp_2m`, `dpmpp_sde`. Must match a sampler available in the deployed bundle. |
| `DEFAULT_SCHEDULER` | `beta` | `str` | Noise schedule. Common values: `beta`, `karras`, `normal`, `simple`. Must be compatible with the chosen sampler. |

---

## Missing Settings

The following settings are **architecturally required** by the planned OAuth2 PKCE / JWT
authentication system but are **not yet present** in `src/core/config.py`. They should be
added before any auth work is merged.

```python
# Recommended additions to Settings in src/core/config.py

# Auth / JWT
secret_key: str = Field(
    description="Secret key for signing JWT access and refresh tokens. "
                "Generate with: python -c \"import secrets; print(secrets.token_hex(32))\". "
                "REQUIRED — no safe default possible.",
)
access_token_expire_minutes: int = Field(
    default=15,
    ge=1,
    description="JWT access token lifetime in minutes.",
)
refresh_token_expire_days: int = Field(
    default=30,
    ge=1,
    le=90,
    description="JWT refresh token lifetime in days.",
)

# CORS (needed once a frontend is served from a different origin)
allowed_origins: list[str] = Field(
    default=["http://localhost:3000"],
    description="List of origins allowed by CORS middleware.",
)
```

Corresponding `.env` entries:

```bash
# =============================================================================
# Authentication (JWT)
# =============================================================================
# Generate: python -c "import secrets; print(secrets.token_hex(32))"
SECRET_KEY=your_64_char_hex_secret_here

ACCESS_TOKEN_EXPIRE_MINUTES=15
REFRESH_TOKEN_EXPIRE_DAYS=30

# =============================================================================
# CORS
# =============================================================================
# Comma-separated list of allowed origins
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
```

> **Security note:** `SECRET_KEY` has no safe fallback value. The service should **refuse to
> start** if it is missing or too short (< 32 bytes), not fall back to a hardcoded string.
> Enforce this with a `@model_validator` in `Settings`.

---

## Environment-Specific Cheatsheets

### Local development (minimal `.env`)

```bash
# Only what differs from defaults
DATABASE_URL=postgresql+asyncpg://apex:apex@localhost:5432/apex
DEBUG=true
DB_ECHO=true

# R2 can be omitted — storage endpoints will be unavailable but the rest works
```

### Production (minimum required set)

```bash
DATABASE_URL=postgresql+asyncpg://apex:strongpassword@db-host:5432/apex

R2_ACCOUNT_ID=abc123...
R2_ACCESS_KEY_ID=...
R2_SECRET_ACCESS_KEY=...
R2_BUCKET_NAME=apex-user-content
R2_PUBLIC_URL_BASE=https://cdn.yourdomain.com

SECRET_KEY=<64-char hex>        # once auth is implemented

COMFYUI_HOST=<gpu-node-ip>

DEBUG=false
DB_ECHO=false
RETENTION_DAYS=7
```
