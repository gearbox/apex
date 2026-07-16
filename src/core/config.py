"""Application configuration using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.core.constants import MAX_EXTRACT_TIMESTAMPS, MAX_PREVIEW_FRAME_COUNT
from src.core.enums import WorkerMode
from src.core.topup_pricing import build_tiers


class GrokBillingConfig(BaseModel):
    """Grok moderation billing policy configuration."""

    moderation_billing_policy: Literal["charge", "refund"] = "charge"
    # "charge" = no refund when respect_moderation=False
    #            (default — conservative, matches xAI's stated text API policy)
    # "refund" = refund when respect_moderation=False
    #            (use if xAI formally confirms no charge for image/video moderation)


# Unsafe placeholder that ships as the default - reject it at startup
_INSECURE_JWT_DEFAULTS: frozenset[str] = frozenset(
    {
        "CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET_KEY_256_BITS",
        "change_me",
        "secret",
        "your-secret-key",
        "your-secure-random-key-here-use-secrets-module",
    }
)
_MIN_JWT_SECRET_BYTES = 32


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Generation & AI service settings
    # -------------------------------------------------------------------------

    # ComfyUI connection settings
    comfyui_host: str = Field(default="127.0.0.1", description="ComfyUI server host")
    comfyui_port: int = Field(default=18188, description="ComfyUI server port")

    # xAI Grok API settings
    xai_api_key: str = Field(
        default="",
        description="xAI API key for Grok image/video generation",
    )
    xai_api_base_url: str = Field(
        default="https://api.x.ai",
        description="xAI API base URL",
    )
    xai_timeout: int = Field(
        default=300,
        ge=30,
        le=900,
        description="xAI API timeout in seconds",
    )

    # Vast.ai API settings
    vastai_api_key: str = Field(
        default="",
        description="Vast.ai API key for GPU node management. Required for Aisha on-demand sessions.",
    )
    vastai_max_429_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description=(
            "Number of times to retry a Vast.ai API call on HTTP 429 "
            "(rate limit), honoring the Retry-After header. After exhaustion, "
            "raises VastAIRateLimitError. Each retry waits Retry-After seconds, "
            "or 1.0s fallback if absent. Zero disables retry."
        ),
    )
    vastai_max_retry_after_seconds: float = Field(
        default=10.0,
        gt=0.0,
        le=60.0,
        description=(
            "Cap on the Retry-After value apex will honor. Defends against a "
            "pathological Retry-After: 300 that would block request handlers "
            "for minutes. If Vast.ai asks for longer than this, retry budget "
            "is consumed immediately at this cap."
        ),
    )
    vastai_destroy_retry_attempts: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Total destroy_instance attempts in _teardown_external_resources "
            "before marking the stop as 'teardown failed' and relying on the "
            "orphan sweeper to recover later. Exponential backoff between attempts: "
            "1s, 2s, 4s, capped at 5s (so 5 attempts wait at most 1+2+4+5 = 12s "
            "in addition to actual API call durations). Each attempt itself may "
            "wait further under HTTP 429 (see vastai_max_429_retries). "
            "Above this attempt count, the orphan sweeper recovers within "
            "orphaned_tunnel_cleanup_interval_minutes."
        ),
    )

    # --- GPU Session Provisioning (ai-bundles) ---
    ai_bundles_github_token: str = Field(
        default="",
        description="GitHub PAT for cloning the private ai-bundles repository",
    )
    ai_bundles_repo_url: str = Field(
        default="https://github.com/gearbox/ai-bundles.git",
        description="Git URL for the ai-bundles repository",
    )
    ai_bundles_sync_interval_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="How often to git-pull ai-bundles for bundle index updates",
    )
    hf_token: str = Field(
        default="",
        description="HuggingFace token for private model downloads on GPU nodes",
    )
    civitai_api_token: str = Field(
        default="",
        description="Civitai API token for model downloads on GPU nodes",
    )
    ai_bundles_branch: str = Field(
        default="master",
        description="Git branch for ai-bundles repository (forwarded to the Aisha CLI as ACS_BUNDLES_BRANCH)",
    )
    ai_bundles_max_download_bytes: int = Field(
        default=50 * 1024 * 1024,  # 50 MB
        ge=1024,
        description=(
            "Maximum compressed tarball size (bytes) accepted from the GitHub "
            "tarball API. The legitimate ai-bundles repo is <1 MB; 50 MB is "
            "1000x headroom. Aborts download mid-stream if exceeded."
        ),
    )
    ai_bundles_max_member_count: int = Field(
        default=5000,
        ge=1,
        description=(
            "Maximum number of entries (files+dirs) accepted in the tarball. "
            "Defends against million-tiny-files inode-exhaustion bombs."
        ),
    )
    ai_bundles_max_member_size_bytes: int = Field(
        default=10 * 1024 * 1024,  # 10 MB
        ge=1024,
        description=(
            "Maximum decompressed size (bytes) for any single tarball member. "
            "Enforced both against TarInfo.size (metadata) and against actual "
            "bytes read during extraction (defends against lying-size headers)."
        ),
    )
    ai_bundles_max_uncompressed_bytes: int = Field(
        default=100 * 1024 * 1024,  # 100 MB
        ge=1024,
        description=(
            "Maximum total decompressed size (bytes) across all tarball members. "
            "Defends against many-medium-files compression-ratio bombs."
        ),
    )

    # --- Aisha CLI runtime ---
    aisha_repo_url: str = Field(
        default="https://github.com/gearbox/aisha.git",
        description="Git URL for the Aisha CLI repository (forwarded to the CLI as ACS_AISHA_REPO)",
    )
    aisha_branch: str = Field(
        default="master",
        description="Git branch for the Aisha CLI repository (forwarded to the CLI as ACS_AISHA_BRANCH)",
    )
    aisha_comfyui_host: str = Field(
        default="0.0.0.0",  # noqa: S104
        description=(
            "ComfyUI bind address on Aisha GPU nodes (forwarded as ACS_COMFYUI_HOST). "
            "Must be 0.0.0.0 in production so cloudflared can reach ComfyUI; deviating "
            "from this default will break the tunnel."
        ),
    )
    aisha_comfyui_extra_args: str = Field(
        default="",
        description=(
            "Extra args appended to ComfyUI launch on Aisha GPU nodes "
            "(forwarded as ACS_COMFYUI_EXTRA_ARGS). Example: '--preview-method auto'."
        ),
    )

    # --- Cloudflare API (per-session GPU node tunnels — NOT the apex API's own cloudflared sidecar) ---
    aisha_cf_api_token: str = Field(
        default="",
        description=(
            "Cloudflare API token with Tunnel:Edit permission for per-session GPU node tunnels "
            "(NOT the staging API's own cloudflared sidecar — that uses CLOUDFLARE_TUNNEL_TOKEN)."
        ),
    )
    aisha_cf_account_id: str = Field(
        default="",
        description=(
            "Cloudflare account ID for per-session GPU node tunnels "
            "(NOT the staging API's own cloudflared sidecar — that uses CLOUDFLARE_TUNNEL_TOKEN)."
        ),
    )
    aisha_cf_zone_id: str = Field(
        default="",
        description=(
            "Cloudflare zone ID for the GPU tunnel domain for per-session GPU node tunnels "
            "(NOT the staging API's own cloudflared sidecar — that uses CLOUDFLARE_TUNNEL_TOKEN)."
        ),
    )
    aisha_cf_tunnel_domain: str = Field(
        default="your-gpu-domain.com",
        description=(
            "CF zone for tunnel DNS records. "
            "Base domain for GPU session tunnel hostnames for per-session GPU node tunnels "
            "(NOT the staging API's own cloudflared sidecar — that uses CLOUDFLARE_TUNNEL_TOKEN)."
            "Tunnel hostnames are constructed as gpu-{session_id_short}.{this}. "
            "Must be a single-level subdomain of the apex zone "
            "so Cloudflare Universal SSL (free plan) covers it."
        ),
    )

    aisha_cf_tunnel_prefix: str | None = Field(
        default="gpu-",
        description=(
            "Required prefix for GPU tunnel hostnames (e.g. 'gpu-'). "
            "Defends against a smuggling attack where a malicious user creates a "
            "hostname like 'abc123.evil.com.your-gpu-domain.com' that passes the "
            "endswith check but actually resolves to evil.com. By enforcing a "
            "prefix, we ensure the hostname structure is {prefix}{session_id}.{allowed_suffix}."
        ),
    )

    # --- GPU Provisioning Worker ---
    gpu_provision_poll_interval_seconds: int = Field(
        default=15,
        ge=5,
        le=60,
        description="How often to poll pending/provisioning/resuming sessions",
    )
    gpu_provision_timeout_seconds: int = Field(
        default=2000,
        ge=300,
        description=(
            "Max wall-clock seconds for a GPU session to finish provisioning (reach ready). "
            "Generous to cover large model downloads (e.g. ~28 GB) on throttled nodes. "
            "Env: GPU_PROVISION_TIMEOUT_SECONDS."
        ),
    )
    gpu_resume_timeout_minutes: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Max time to wait for a paused session to resume",
    )
    provisioning_offer_walk_depth: int = Field(
        default=10,
        ge=1,
        le=20,
        description=(
            "Within a single provisioning attempt, how many cheapest offers "
            "to walk past on OfferTakenError before declaring this attempt "
            "failed. Cheap — each attempt is a quick API call. Default 10 "
            "balances 'try a few backup offers' against 'don't waste tokens "
            "on a dead inventory'."
        ),
    )
    provisioning_recreation_attempts: int = Field(
        default=1,
        ge=1,
        le=3,
        description=(
            "Total provisioning attempts before the session transitions to "
            "'failed' terminally. Each attempt creates a new Vast.ai node "
            "with fresh offers. Default 1 means 'no autonomous retry' — "
            "if the first attempt fails, the user must start a new session. "
            "This prevents unattended retries from racking up GPU billing "
            "after the user has given up."
        ),
    )
    vastai_offer_search_limit: int = Field(
        default=20,
        ge=5,
        le=50,
        description=(
            "How many offers to fetch from Vast.ai search. Larger than "
            "provisioning_offer_walk_depth so cooldown filtering cannot starve "
            "the offer walk."
        ),
    )
    node_cooldown_base_minutes: int = Field(
        default=30,
        ge=1,
        le=1440,
        description=(
            "Initial cooldown for a Vast.ai machine after a provisioning failure. "
            "Doubles per repeated failure within the failure window, capped at "
            "node_cooldown_max_minutes. Set equal to the max to get a fixed TTL."
        ),
    )
    node_cooldown_max_minutes: int = Field(
        default=360,
        ge=1,
        le=4320,
        description="Upper bound for the escalating machine cooldown TTL.",
    )
    node_cooldown_failure_window_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description=(
            "Sliding window for the per-machine failure counter that drives TTL "
            "escalation. Each new failure refreshes the window."
        ),
    )

    # --- Orphaned Tunnel Cleanup Worker ---
    orphaned_tunnel_cleanup_interval_minutes: int = Field(
        default=60,
        ge=5,
        le=1440,
        description="How often to run the orphaned tunnel cleanup sweep",
    )
    orphaned_tunnel_cleanup_grace_period_minutes: int = Field(
        default=10,
        ge=1,
        le=120,
        description="Tunnels younger than this are skipped by the orphan cleanup worker",
    )
    orphaned_instance_cleanup_grace_period_minutes: int = Field(
        default=5,
        ge=1,
        le=60,
        description=(
            "Skip sessions stopped/failed within this window — gives the in-flow "
            "stop logic time to complete naturally before the sweeper interferes."
        ),
    )
    orphaned_instance_cleanup_horizon_minutes: int = Field(
        default=1440,
        ge=60,
        le=10080,
        description=(
            "Don't query Vast.ai for orphan candidates older than this. After "
            "24h, an instance that wasn't destroyed has almost certainly been "
            "garbage-collected by Vast.ai. Bounded to keep the orphan sweep cost "
            "constant as the gpu_sessions table grows."
        ),
    )
    gpu_provision_worker_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max sessions processed concurrently per provisioning worker sweep",
    )
    gpu_provision_stall_timeout_seconds: int = Field(
        default=420,
        ge=60,
        le=1800,
        description=(
            "Seconds of silence before the worker treats a provisioning session as stalled. "
            "Only activates when callbacks are flowing (last_progress_at is set); falls back to "
            "gpu_provision_timeout_seconds when no callbacks have ever arrived. "
            "A stalled session is retryable (unlike a ceiling timeout). "
            "Default 420s (7 min) is generous vs. the node's ~3s throttle. "
            "Env: GPU_PROVISION_STALL_TIMEOUT_SECONDS."
        ),
    )

    # --- Billing Finalization Reconciler ---
    billing_reconciler_interval_minutes: int = Field(
        default=10,
        ge=1,
        le=120,
        description=(
            "How often the billing reconciler scans for sessions where "
            "billing_finalized_at IS NULL AND status='stopped'. Shorter than "
            "the orphan cleanup interval (default 60min) because a stuck "
            "billing finalization means a user is potentially over- or "
            "under-charged until reconciled."
        ),
    )
    billing_reconciler_grace_period_minutes: int = Field(
        default=2,
        ge=1,
        le=30,
        description=(
            "Sessions whose stopped_at is younger than this are skipped. "
            "Avoids racing with the in-line _finalize_billing retry path "
            "which can take up to ~2s but may be longer under load."
        ),
    )
    billing_reconciler_quarantine_threshold: int = Field(
        default=10,
        ge=1,
        le=100,
        description=(
            "After this many failed reconciliation attempts, the worker logs "
            "at ERROR with a quarantine flag for ops alerting. The session "
            "row is NOT mutated — billing_finalized_at stays NULL so the "
            "worker keeps retrying after the underlying issue is fixed."
        ),
    )
    billing_reconciler_max_per_sweep: int = Field(
        default=50,
        ge=1,
        le=500,
        description=(
            "Maximum sessions processed per sweep. Caps wall-clock time of "
            "any single sweep so a backlog can't starve the worker loop."
        ),
    )

    # --- GPU Session Billing ---
    gpu_session_tokens_per_minute: int = Field(
        default=100,
        ge=1,
        description=(
            "Token cost per active minute of GPU session runtime. "
            "The start-time reservation is computed as MIN_BILLABLE_MINUTES * rate "
            "(see src.api.services.gpu_session.service._MIN_BILLABLE_MINUTES). "
            "Keeping reservation and rate coupled prevents settings drift from "
            "causing over-refunds between session start and finalize."
        ),
    )

    gpu_session_stale_grace_seconds: int = Field(
        default=120,
        ge=30,
        le=600,
        description=(
            "Seconds a session may be unreachable before the health reconciler marks it stale. "
            "Must be ≥ 2× the reconciler cycle (60s default) to absorb a single transient blip. "  # noqa: RUF001
            "last_reachable_at is stamped at activation and on every healthy probe."
        ),
    )

    # --- GPU Session Credit Guard ---
    gpu_session_credit_safety_factor: float = Field(
        default=1.5,
        ge=1.0,
        description=(
            "Multiplier applied to the per-cycle token cost to derive the floor_tokens threshold. "
            "A session is terminated when balance ≤ floor_tokens. "
            "Higher values (e.g. 2.0) trigger earlier termination to avoid billing in arrears."
        ),
    )
    gpu_session_credit_warning_minutes: int = Field(
        default=20,
        ge=1,
        description=(
            "Balance threshold (in equivalent minutes of GPU time) below which a WARNING "
            "credit event is emitted. Must be > gpu_session_credit_critical_minutes."
        ),
    )
    gpu_session_credit_critical_minutes: int = Field(
        default=10,
        ge=1,
        description=(
            "Balance threshold (in equivalent minutes of GPU time) below which a CRITICAL "
            "credit event is emitted. Must be < gpu_session_credit_warning_minutes."
        ),
    )

    # --- Phase 2 Callback (pre-wired) ---
    apex_callback_url: str = Field(
        default="",
        description="Public URL for GPU node → Apex callbacks (Phase 2, unused in Phase 1)",
    )

    # Grok video polling settings
    grok_video_poll_interval: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Seconds between video status polls",
    )
    grok_video_max_poll_time: int = Field(
        default=600,
        ge=60,
        le=1800,
        description="Maximum seconds to poll for video completion",
    )
    grok_video_worker_in_process: bool = Field(
        default=False,
        description=(
            "Start Grok video polling worker in-process. "
            "Set to True for single-worker deployments. "
            "For multi-worker Granian, use the standalone worker instead."
        ),
    )
    grok_video_max_concurrent_polls: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Maximum concurrent xAI status polls per Grok video worker tick.",
    )

    # Generation defaults — bundle.yaml overrides these per-model when present
    default_steps: int = Field(default=12, description="Default generation steps")
    max_steps: int = Field(default=20, description="Maximum generation steps")
    default_cfg: float = Field(default=1.1, description="Default CFG scale")
    default_sampler: str = Field(default="euler", description="Default sampler")
    default_scheduler: str = Field(default="beta", description="Default scheduler")
    default_denoise: float = Field(default=1.0, description="Default denoise strength")
    default_resolution: str = Field(
        default="standard",
        description="Default image resolution tier (bundle.yaml overrides per-model)",
    )

    # -------------------------------------------------------------------------
    # System settings
    # -------------------------------------------------------------------------

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API server host")  # noqa: S104
    api_port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")
    enable_docs: bool = Field(
        default=False,
        description=(
            "Expose the OpenAPI schema and Swagger UI at /docs. Off by default; "
            "typically enabled only in dev/staging. Never enable in production."
        ),
    )
    build_sha: str = Field(
        default="unknown",
        description="Git SHA of the running build, injected at image build time.",
    )

    # ASGI Granian settings
    asgi_workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description=(
            "Number of worker processes for ASGI server. Increase in production based on "
            "CPU cores and expected load. Values > 1 require REDIS_URL (for worker leader "
            "election) unless WORKER_MODE=api_only."
        ),
    )

    # -------------------------------------------------------------------------
    # Background workers
    # -------------------------------------------------------------------------
    worker_mode: WorkerMode = Field(
        default=WorkerMode.all,
        description=(
            "Which in-process background workers this process runs. 'all' (default) starts "
            "every worker alongside the API. 'api_only' starts none — use for API-only "
            "Granian processes when workers run elsewhere."
        ),
    )

    # App URL (used for building verification / reset links)
    app_url: str = Field(
        default="http://localhost:3000",
        description=(
            "Base URL of the frontend application. "
            "Used to build verification and password reset links. "
            "Must be set to the real frontend URL in production."
        ),
    )

    # JWT Authentication Settings
    jwt_secret_key: str = Field(
        default="CHANGE_ME_IN_PRODUCTION_USE_STRONG_SECRET_KEY_256_BITS",
        description=(
            "Secret key for JWT signing. "
            'Generate with: python -c "import secrets; print(secrets.token_urlsafe(32))". '
            "Must be at least 32 bytes. No safe default — must be set explicitly."
        ),
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    jwt_access_token_expire_minutes: int = Field(
        default=15,
        ge=1,
        le=60,
        description="Access token expiration in minutes",
    )
    jwt_refresh_token_expire_days: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Refresh token expiration in days",
    )
    jwt_issuer: str | None = Field(
        default="apex-api",
        description="JWT issuer claim",
    )

    # Content cookie
    content_cookie_ttl_hours: int = Field(
        default=1,
        ge=1,
        le=24,
        description=(
            "TTL in hours for the content authentication cookie (apex_content). "
            "Kept short so a stateless token's post-revocation window is small; "
            "the cookie is refreshed on every login/token refresh, so this never "
            "expires mid-session."
        ),
    )
    content_cookie_name: str = Field(
        default="apex_content",
        description="Name of the content authentication cookie.",
    )

    @property
    def content_cookie_secure(self) -> bool:
        """Secure flag for the content cookie — disabled in debug/dev mode."""
        return not self.debug

    # Logging settings
    log_level: str = Field(
        default="INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    log_format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format: 'json' for production, 'console' for development",
    )

    # Worker Settings
    token_cleanup_interval_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Seconds between expired token cleanup runs.",
    )

    # Idempotency settings
    idempotency_key_ttl_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="TTL in hours for idempotency key records before cleanup",
    )
    idempotency_processing_stale_seconds: int = Field(
        default=120,
        ge=10,
        description=(
            "Seconds a 'processing' idempotency record may sit unresolved before "
            "a retry with the same key is allowed to reclaim it — the owning "
            "request is presumed dead (process crash, poisoned session, etc.). "
            "Must exceed the worst-case legitimate submit time (ComfyUI queue + "
            "tunnel handshake, Stripe/NowPayments round trip). Below this "
            "threshold, a same-key retry gets 409 (a genuinely concurrent request)."
        ),
    )

    # Health check
    health_check_timeout_seconds: float = Field(
        default=15.0,
        ge=1.0,
        le=30.0,
        description=(
            "Per-component timeout in seconds for health checks. "
            "Must be > GPU probe timeout (10s) to avoid false timeouts on reconciler."
        ),
    )
    health_snapshot_interval_seconds: int = Field(
        default=60,
        ge=10,
        le=600,
        description=(
            "Interval in seconds between health snapshot persistence cycles. "
            "Each cycle: run all checks → write to DB → publish via Redis."
        ),
    )
    health_snapshot_retention_days: int = Field(
        default=30,
        ge=1,
        le=365,
        description="Days to retain health snapshots before cleanup.",
    )

    # Rate Limiting
    redis_url: str | None = Field(
        default=None,
        description=(
            "Redis URL for rate limiting and pub/sub, e.g. redis://localhost:6379/0. "
            "When None, falls back to in-memory storage for rate limiting (single-process only) "
            "and SSE/pub-sub will not function. Recommended: set to redis://redis:6379/0 in "
            "production Docker environments."
        ),
    )
    trusted_ip_header: Literal["cf-connecting-ip", "x-forwarded-for", "none"] = Field(
        default="none",
        description=(
            "Header trusted for client-IP-based rate limiting. 'none' (default, dev) "
            "uses only the ASGI connection address — never a client-controlled header. "
            "'cf-connecting-ip' for Cloudflare-fronted deployments (CF strips/overwrites "
            "this header at the edge, so it's unforgeable provided the origin only "
            "accepts CF traffic). 'x-forwarded-for' for a generic reverse proxy — "
            "combine with trusted_proxy_hops and never trust the leftmost entry, which "
            "is client-controlled."
        ),
    )
    trusted_proxy_hops: int = Field(
        default=0,
        ge=0,
        description=(
            "Only used when trusted_ip_header='x-forwarded-for'. Number of trusted "
            "reverse-proxy hops — selects the X-Forwarded-For entry at position "
            "(count - 1 - trusted_proxy_hops), i.e. rightmost-minus-hops. 0 means "
            "trust the rightmost (last-appended) entry, which is the proxy closest "
            "to this service."
        ),
    )
    rate_limit_sse_ticket: str = Field(
        default="10/minute",
        description="SSE ticket issuance endpoint rate limit.",
    )

    # SSE settings
    sse_ticket_ttl_seconds: int = Field(
        default=30,
        ge=5,
        le=120,
        description="TTL for one-time SSE connection tickets.",
    )
    sse_heartbeat_interval: int = Field(
        default=15,
        ge=5,
        le=60,
        description="Seconds between SSE keepalive comments.",
    )
    rate_limit_register: str = Field(
        default="5/hour",
        description="Register endpoint rate limit.",
    )
    rate_limit_login: str = Field(
        default="10/minute",
        description="Login endpoint rate limit.",
    )
    rate_limit_forgot_password: str = Field(
        default="3/hour",
        description="Forgot-password endpoint rate limit.",
    )
    rate_limit_resend_verification: str = Field(
        default="3/hour",
        description="Resend-verification endpoint rate limit.",
    )

    # --------------------------------------------------------------------------
    # Storage & External Services
    # --------------------------------------------------------------------------

    # Database settings
    database_url: str = Field(
        default="postgresql+asyncpg://apex:apex@localhost:5432/apex",
        description="PostgreSQL connection URL (async format)",
    )
    db_pool_size: int = Field(default=5, description="Database connection pool size")
    db_max_overflow: int = Field(default=10, description="Max overflow connections")
    db_echo: bool = Field(default=False, description="Echo SQL statements")

    # Cloudflare R2 settings
    r2_account_id: str = Field(
        default="",
        description="Cloudflare R2 account ID",
    )
    r2_access_key_id: str = Field(
        default="",
        description="R2 access key ID",
    )
    r2_secret_access_key: str = Field(
        default="",
        description="R2 secret access key",
    )
    r2_bucket_name: str = Field(
        default="apex-user-content",
        description="R2 bucket name for user content",
    )
    r2_public_url_base: str | None = Field(
        default=None,
        description="Public URL base for R2 content (if using custom domain)",
    )
    r2_public_assets_bucket: str | None = Field(
        default=None,
        description=(
            "Dedicated public R2 bucket for static assets (currency logos, etc.) — distinct "
            "from r2_bucket_name, which is the private, cookie-authenticated user-content "
            "bucket. Never make r2_bucket_name public; use this bucket for anything meant to "
            "be served directly. Both this and r2_public_assets_url_base must be set to "
            "enable logo caching."
        ),
    )
    r2_public_assets_url_base: str | None = Field(
        default=None,
        description=(
            "Public URL base (custom domain) fronting r2_public_assets_bucket. Both this and "
            "r2_public_assets_bucket must be set to enable logo caching."
        ),
    )

    content_url_ttl: int = Field(
        default=10800,
        ge=60,
        le=86400,
        description="Cache-Control max-age for content proxy responses (seconds).",
    )

    # Storage retention settings
    retention_days: int = Field(
        default=7,
        ge=1,
        le=90,
        description="Days to retain user content before cleanup",
    )
    content_cleanup_enabled: bool = Field(
        default=True,
        description="Enable the periodic content retention sweeper.",
    )
    content_cleanup_interval_seconds: int = Field(
        default=900,
        ge=60,
        le=86400,
        description="Seconds between content retention sweep runs.",
    )
    content_cleanup_batch_size: int = Field(
        default=500,
        ge=1,
        le=5000,
        description="Max expired rows fetched per sweep batch (per content type).",
    )
    content_cleanup_max_batches_per_run: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Upper bound on batches drained in a single sweep run (backlog control).",
    )
    max_upload_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum upload file size in MB",
    )
    image_max_input_megapixels: float = Field(
        default=100.0,
        gt=0,
        description=(
            "Maximum decoded pixel count (in megapixels) for input images, "
            "enforced before full decode in image_normalization.py. Guards "
            "against decompression-bomb uploads and i2i remix inputs."
        ),
    )

    # -------------------------------------------------------------------------
    # Video frame extraction
    # -------------------------------------------------------------------------
    frame_extract_max_video_seconds: int = Field(
        default=300,
        ge=1,
        le=3600,
        description="Upload-time ffprobe rejection cap for uploaded video duration.",
    )
    frame_extract_ffmpeg_timeout_seconds: int = Field(
        default=30,
        ge=1,
        le=300,
        description="Wall-clock timeout for each ffmpeg/ffprobe subprocess invocation.",
    )
    frame_extract_poll_interval_seconds: float = Field(
        default=2.0,
        ge=0.5,
        le=60.0,
        description="Seconds between FrameExtractionWorker queue polls.",
    )
    frame_preview_max_edge: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Longest-edge pixel cap for preview strip WEBP frames.",
    )
    frame_preview_url_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        le=86400,
        description="Presigned URL TTL for preview frames, generated per-read.",
    )
    frame_preview_retention_days: int = Field(
        default=2,
        ge=1,
        le=30,
        description=(
            "Documented lifecycle-rule value for the 'frame-previews/' R2 prefix. "
            "Informational only — enforcement is an R2 lifecycle rule configured "
            "out-of-band, not this application."
        ),
    )
    frame_extract_stale_running_seconds: int = Field(
        default=1800,
        ge=300,
        le=7200,
        description=(
            "A frame_extraction_jobs row stuck in 'running' longer than this is "
            "assumed to be from a worker that died mid-execution and is failed by "
            "the next sweep. Must be >= max(MAX_PREVIEW_FRAME_COUNT, "
            "MAX_EXTRACT_TIMESTAMPS) * frame_extract_ffmpeg_timeout_seconds — a "
            "legitimate job can run that long, and the sweep must never fire on a "
            "live job. Enforced by a model validator."
        ),
    )

    # Email
    resend_api_key: str = Field(
        default="",
        description=(
            "Resend API key (re_...). "
            "When empty, LogEmailService is used — emails are logged to stdout only."
        ),
    )
    email_from_address: str = Field(
        default="noreply@apex.ai",
        description="Sender email address. Must be verified in your Resend dashboard.",
    )
    email_from_name: str = Field(
        default="Apex",
        description="Sender display name shown in email clients.",
    )

    # -------------------------------------------------------------------------
    # Billing & Payment settings
    # -------------------------------------------------------------------------

    # Billing & Payment settings
    grok_moderation_billing_policy: Literal["charge", "refund"] = Field(
        default="charge",
        description="Policy for Grok moderation: 'charge' or 'refund'",
    )

    # Stripe (legacy single-product — kept for backward compatibility)
    stripe_secret_key: str = Field(
        default="",
        description="Stripe secret API key (legacy). Prefer per-product keys.",
    )
    stripe_webhook_secret: str = Field(
        default="",
        description="Stripe webhook endpoint signing secret (legacy). Prefer per-product keys.",
    )

    # Per-product Stripe keys
    stripe_secret_key_vex: str | None = Field(
        default=None,
        description="Stripe secret key for vex.pics",
    )
    stripe_webhook_secret_vex: str | None = Field(
        default=None,
        description="Stripe webhook secret for vex.pics",
    )
    stripe_publishable_key_vex: str | None = Field(
        default=None,
        description="Stripe publishable key for vex.pics",
    )
    stripe_secret_key_synthara: str | None = Field(
        default=None,
        description="Stripe secret key for Synthara",
    )
    stripe_webhook_secret_synthara: str | None = Field(
        default=None,
        description="Stripe webhook secret for Synthara",
    )
    stripe_publishable_key_synthara: str | None = Field(
        default=None,
        description="Stripe publishable key for Synthara",
    )

    # NowPayments (legacy single-product — kept for backward compatibility)
    nowpayments_api_key: str = Field(
        default="",
        description="NowPayments API key (legacy). Prefer per-product keys.",
    )
    nowpayments_ipn_secret: str = Field(
        default="",
        description="NowPayments IPN HMAC secret (legacy). Prefer per-product keys.",
    )
    nowpayments_api_base: str = Field(
        default="https://api.nowpayments.io",
        description="NowPayments API base URL; override to use the sandbox.",
    )

    # Per-product NowPayments keys (vex.pics only — crypto payments for consumers)
    nowpayments_api_key_vex: str | None = Field(
        default=None,
        description="NowPayments API key for vex.pics",
    )
    nowpayments_ipn_secret_vex: str | None = Field(
        default=None,
        description="NowPayments IPN HMAC secret for vex.pics",
    )

    # Stripe Checkout redirect paths — appended to a product's ProductConfig.frontend_origin.
    # Kept as settings (not hardcoded) so ops can repoint them without a backend deploy.
    stripe_checkout_success_path: str = Field(
        default="/billing?success=true",
        description="Path+query appended to frontend_origin for the Stripe success redirect.",
    )
    stripe_checkout_cancel_path: str = Field(
        default="/billing?cancelled=true",
        description="Path+query appended to frontend_origin for the Stripe cancel redirect.",
    )

    # Tiered free-amount top-up
    billing_pricing_tiers: dict[int, int] = Field(
        default={10: 0, 50: 0, 100: 5, 250: 10},
        description=(
            "Top-up discount tiers: threshold_usd -> discount_pct. An amount qualifies for "
            "the highest threshold at or below it. Env: BILLING_PRICING_TIERS as a JSON "
            'object, e.g. \'{"10":0,"50":0,"100":5,"250":10}\'.'
        ),
    )
    billing_tokens_per_usd: int = Field(
        default=100,
        gt=0,
        description=(
            "Tokens granted per USD of credits (pre-discount), derived from the former "
            "starter package rate (1,000 tokens / $9.99 ~= 100)."
        ),
    )
    billing_min_topup_usd: int = Field(
        default=5,
        gt=0,
        description="Minimum top-up amount in whole USD.",
    )
    billing_max_topup_usd: int = Field(
        default=10_000,
        gt=0,
        description="Maximum top-up amount in whole USD.",
    )

    # --- Payment Currency Catalog Sync ---
    payment_currency_sync_interval_seconds: int = Field(
        default=10800,
        ge=300,
        le=86400,
        description=(
            "Seconds between periodic payment currency catalog syncs (merchant/coins "
            "+ full-currencies + logo caching). Default 3h matches how rarely a "
            "NowPayments dashboard's enabled-currency list actually changes; also "
            "triggerable on-demand via POST /v1/admin/payments/currencies/refresh."
        ),
    )

    # -------------------------------------------------------------------------
    # Branding & Product Segmentation Settings
    # -------------------------------------------------------------------------

    # Branding
    app_name: str = Field(
        default="Apex",
        description=(
            "Public-facing product name used in emails and UI copy. "
            "Override via APP_NAME env var when the brand/domain changes."
        ),
    )
    support_email: str = Field(
        default="support@apex.ai",
        description="Support email address shown in transactional emails.",
    )

    # Product
    default_product: str = Field(
        default="vex",
        description="Default product slug for localhost/dev requests.",
    )

    # Per-product app URLs (for email links)
    app_url_vex: str = Field(
        default="https://vex.pics",
        description="Base URL for vex.pics frontend (used in email links).",
    )
    app_url_synthara: str = Field(
        default="https://synthara.app",
        description="Base URL for Synthara frontend (used in email links).",
    )

    # -------------------------------------------------------------------------
    # Aisha Job Poller
    # -------------------------------------------------------------------------

    aisha_poller_enabled: bool = Field(
        default=True,
        description=(
            "Enable the Aisha job polling worker. Set to false to disable polling "
            "(useful for tests and ops kill-switch)."
        ),
    )
    aisha_poller_tick_interval_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Seconds between poller ticks.",
    )
    aisha_poller_max_concurrent_polls: int = Field(
        default=16,
        ge=1,
        le=256,
        description="Maximum concurrent ComfyUI polls per tick.",
    )
    aisha_poller_job_age_warning_seconds: int = Field(
        default=300,
        ge=30,
        description="Log a warning if a job has been running longer than this.",
    )
    aisha_poller_job_age_timeout_seconds: int = Field(
        default=1800,
        ge=60,
        description="Mark a job FAILED if it has been running longer than this.",
    )
    aisha_poller_comfyui_request_timeout_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="Per-request timeout for ComfyUI API calls from the poller.",
    )

    # -------------------------------------------------------------------------
    # Web Push Notifications
    # -------------------------------------------------------------------------

    vapid_public_key: str | None = Field(
        default=None,
        description=(
            "VAPID public key (base64url, uncompressed EC point) for Web Push. "
            "Generate with tools/generate_vapid_keys.py. Push is enabled only when "
            "this, vapid_private_key, and vapid_subject are all set AND redis_url "
            "is configured."
        ),
    )
    vapid_private_key: str | None = Field(
        default=None,
        description=(
            "VAPID private key (base64url, raw 32-byte scalar) for Web Push. "
            "Never commit this value — generate with tools/generate_vapid_keys.py "
            "and set via environment/secrets only."
        ),
    )
    vapid_subject: str | None = Field(
        default=None,
        description=(
            "VAPID 'sub' claim identifying the sending application, e.g. "
            "'mailto:ops@apex.ai'. Required by the Web Push protocol so push "
            "services can contact the operator about a misbehaving sender."
        ),
    )
    push_broadcast_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max concurrent Web Push sends per broadcast fan-out batch.",
    )

    # -------------------------------------------------------------------------
    # Validators
    # -------------------------------------------------------------------------

    @model_validator(mode="after")
    def validate_credit_warning_thresholds(self) -> "Settings":
        """Assert critical threshold is strictly below warning threshold."""
        if self.gpu_session_credit_critical_minutes >= self.gpu_session_credit_warning_minutes:
            raise ValueError(
                "gpu_session_credit_critical_minutes must be less than "
                "gpu_session_credit_warning_minutes "
                f"(got critical={self.gpu_session_credit_critical_minutes}, "
                f"warning={self.gpu_session_credit_warning_minutes})"
            )
        return self

    @model_validator(mode="after")
    def validate_jwt_secret_key(self) -> "Settings":
        """Reject insecure JWT secret keys at startup.

        Prevents the service from running with the shipped placeholder value
        or any key that is too short to provide adequate security.

        Raises:
            ValueError: If the key is a known placeholder or shorter than
                        _MIN_JWT_SECRET_BYTES bytes.
        """
        key = self.jwt_secret_key

        if key in _INSECURE_JWT_DEFAULTS:
            raise ValueError(
                "JWT_SECRET_KEY is set to an insecure placeholder. "
                "Generate a strong key with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(32))"\n'
                "and set it in your .env file or environment."
            )

        if len(key.encode()) < _MIN_JWT_SECRET_BYTES:
            raise ValueError(
                f"JWT_SECRET_KEY must be at least {_MIN_JWT_SECRET_BYTES} bytes "
                f"(got {len(key.encode())} bytes). "
                "Generate a strong key with:\n"
                '  python -c "import secrets; print(secrets.token_urlsafe(32))"'
            )

        return self

    @model_validator(mode="after")
    def validate_multi_worker_requires_redis(self) -> "Settings":
        """Reject asgi_workers > 1 without Redis unless workers are disabled.

        Multiple Granian processes each start every worker loop; without Redis
        there is no leader election, so every process would double-provision
        GPU nodes, double-poll jobs, etc. WORKER_MODE=api_only sidesteps this
        entirely by not starting any worker loop.
        """
        if self.asgi_workers > 1 and self.worker_mode is WorkerMode.all and self.redis_url is None:
            raise ValueError(
                "asgi_workers > 1 requires REDIS_URL for worker leader election, "
                "or set WORKER_MODE=api_only"
            )
        return self

    @model_validator(mode="after")
    def validate_node_cooldown_thresholds(self) -> "Settings":
        """Assert the base cooldown never exceeds the cap.

        Equal base/max is allowed and degrades the escalating TTL to a fixed one
        (see node_cooldown_base_minutes description) — only base > max is rejected.
        """
        if self.node_cooldown_base_minutes > self.node_cooldown_max_minutes:
            raise ValueError(
                "node_cooldown_base_minutes must be <= node_cooldown_max_minutes "
                f"(got base={self.node_cooldown_base_minutes}, "
                f"max={self.node_cooldown_max_minutes})"
            )
        return self

    @model_validator(mode="after")
    def validate_frame_extract_stale_running_seconds(self) -> "Settings":
        """Assert the stale-sweep threshold exceeds worst-case job runtime.

        A legitimate frame extraction job can take up to
        max(MAX_PREVIEW_FRAME_COUNT, MAX_EXTRACT_TIMESTAMPS) *
        frame_extract_ffmpeg_timeout_seconds to finish. If the stale threshold is
        lower than that, the sweep can mark a live, still-running job 'failed' —
        a transiently wrong status served to polling clients. Fail loud at
        startup rather than let this surface as flapping job statuses in prod.
        """
        worst_case_seconds = (
            max(MAX_PREVIEW_FRAME_COUNT, MAX_EXTRACT_TIMESTAMPS)
            * self.frame_extract_ffmpeg_timeout_seconds
        )
        if self.frame_extract_stale_running_seconds < worst_case_seconds:
            raise ValueError(
                "frame_extract_stale_running_seconds must be >= "
                "max(MAX_PREVIEW_FRAME_COUNT, MAX_EXTRACT_TIMESTAMPS) * "
                "frame_extract_ffmpeg_timeout_seconds "
                f"(got stale={self.frame_extract_stale_running_seconds}, "
                f"worst_case={worst_case_seconds})"
            )
        return self

    @model_validator(mode="after")
    def validate_billing_pricing_tiers(self) -> "Settings":
        """Fail loud on a malformed top-up pricing config.

        An operator-configured pricing table must never be silently coerced —
        a typo here directly determines what customers are charged.
        """
        tiers = build_tiers(self.billing_pricing_tiers)

        if not tiers:
            raise ValueError("billing_pricing_tiers must not be empty")

        prev_discount = -1
        for tier in tiers:
            if tier.threshold_usd <= 0:
                raise ValueError(
                    f"billing_pricing_tiers thresholds must be positive (got {tier.threshold_usd})"
                )
            if not 0 <= tier.discount_pct <= 90:
                raise ValueError(
                    "billing_pricing_tiers discounts must be in [0, 90] "
                    f"(got {tier.discount_pct} at threshold {tier.threshold_usd})"
                )
            if tier.discount_pct < prev_discount:
                raise ValueError(
                    "billing_pricing_tiers discounts must be non-decreasing with ascending "
                    "thresholds (prevents split-payment arbitrage from a config typo)"
                )
            prev_discount = tier.discount_pct

        if self.billing_min_topup_usd >= self.billing_max_topup_usd:
            raise ValueError(
                "billing_min_topup_usd must be less than billing_max_topup_usd "
                f"(got min={self.billing_min_topup_usd}, max={self.billing_max_topup_usd})"
            )

        lowest_threshold = tiers[0].threshold_usd
        if lowest_threshold < self.billing_min_topup_usd:
            raise ValueError(
                "billing_min_topup_usd must be <= the lowest configured tier threshold "
                f"(got min={self.billing_min_topup_usd}, lowest threshold={lowest_threshold})"
            )

        return self

    # -------------------------------------------------------------------------
    # Computed fields
    # -------------------------------------------------------------------------

    @computed_field  # type: ignore[prop-decorator]
    @property
    def comfyui_base_url(self) -> str:
        """Construct ComfyUI base URL."""
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def r2_endpoint_url(self) -> str:
        """Construct R2 endpoint URL."""
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def r2_configured(self) -> bool:
        """Check if R2 is properly configured."""
        return bool(self.r2_account_id and self.r2_access_key_id and self.r2_secret_access_key)

    @property
    def r2_public_assets_configured(self) -> bool:
        """Check if the dedicated public assets bucket (for currency logos) is configured."""
        return bool(self.r2_public_assets_bucket and self.r2_public_assets_url_base)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def grok_configured(self) -> bool:
        """Check if Grok API is configured."""
        return bool(self.xai_api_key)

    @property
    def vastai_configured(self) -> bool:
        """Check if Vast.ai API is configured."""
        return bool(self.vastai_api_key)

    @property
    def aisha_cf_configured(self) -> bool:
        """Check if Cloudflare API is configured for per-session GPU node tunnels."""
        return bool(self.aisha_cf_api_token and self.aisha_cf_account_id and self.aisha_cf_zone_id)

    @property
    def push_enabled(self) -> bool:
        """Check if Web Push is configured: all three VAPID fields plus Redis."""
        return bool(
            self.vapid_public_key
            and self.vapid_private_key
            and self.vapid_subject
            and self.redis_url
        )

    @property
    def grok_billing(self) -> GrokBillingConfig:
        """Get Grok billing configuration."""
        return GrokBillingConfig(
            moderation_billing_policy=self.grok_moderation_billing_policy,
        )

    @property
    def stripe_configured(self) -> bool:
        """Check if Stripe is configured for at least one product."""
        per_product = bool(
            (self.stripe_secret_key_vex and self.stripe_webhook_secret_vex)
            or (self.stripe_secret_key_synthara and self.stripe_webhook_secret_synthara)
        )
        legacy = bool(self.stripe_secret_key and self.stripe_webhook_secret)
        return per_product or legacy

    @property
    def nowpayments_configured(self) -> bool:
        """Check if NowPayments is configured for at least one product."""
        per_product = bool(self.nowpayments_api_key_vex and self.nowpayments_ipn_secret_vex)
        legacy = bool(self.nowpayments_api_key and self.nowpayments_ipn_secret)
        return per_product or legacy

    def stripe_secret_key_for(self, product_id: str) -> str:
        """Resolve the Stripe secret key for a product, falling back to the legacy key.

        Explicit per-product dict (not getattr reflection) so an unrecognized
        product_id can never probe an unrelated attribute name.

        Raises:
            RuntimeError: Neither a per-product nor a legacy key is configured.
        """
        per_product = {
            "vex": self.stripe_secret_key_vex,
            "synthara": self.stripe_secret_key_synthara,
        }.get(product_id)
        if key := per_product or self.stripe_secret_key:
            return key
        raise RuntimeError(f"No Stripe secret key configured for product '{product_id}'")

    def stripe_webhook_secret_for(self, product_id: str) -> str:
        """Resolve the Stripe webhook signing secret for a product.

        Raises:
            RuntimeError: Neither a per-product nor a legacy secret is configured.
        """
        per_product = {
            "vex": self.stripe_webhook_secret_vex,
            "synthara": self.stripe_webhook_secret_synthara,
        }.get(product_id)
        if secret := per_product or self.stripe_webhook_secret:
            return secret
        raise RuntimeError(f"No Stripe webhook secret configured for product '{product_id}'")

    def nowpayments_api_key_for(self, product_id: str) -> str:
        """Resolve the NowPayments API key for a product (vex.pics only today).

        Raises:
            RuntimeError: Neither a per-product nor a legacy key is configured.
        """
        per_product = {"vex": self.nowpayments_api_key_vex}.get(product_id)
        if key := per_product or self.nowpayments_api_key:
            return key
        raise RuntimeError(f"No NowPayments API key configured for product '{product_id}'")

    def nowpayments_ipn_secret_for(self, product_id: str) -> str:
        """Resolve the NowPayments IPN HMAC secret for a product.

        Raises:
            RuntimeError: Neither a per-product nor a legacy secret is configured.
        """
        per_product = {"vex": self.nowpayments_ipn_secret_vex}.get(product_id)
        if secret := per_product or self.nowpayments_ipn_secret:
            return secret
        raise RuntimeError(f"No NowPayments IPN secret configured for product '{product_id}'")

    @property
    def email_configured(self) -> bool:
        """Return True when a real email provider is configured."""
        return bool(self.resend_api_key)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reset_settings() -> None:
    """Reset cached settings (useful for testing)."""
    get_settings.cache_clear()
