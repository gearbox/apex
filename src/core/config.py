"""Application configuration using pydantic-settings."""

import dataclasses
from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic import BaseModel, Field, computed_field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class GrokBillingConfig(BaseModel):
    """Grok moderation billing policy configuration."""

    moderation_billing_policy: Literal["charge", "refund"] = "charge"
    # "charge" = no refund when respect_moderation=False
    #            (default — conservative, matches xAI's stated text API policy)
    # "refund" = refund when respect_moderation=False
    #            (use if xAI formally confirms no charge for image/video moderation)


@dataclasses.dataclass(frozen=True)
class TokenPackage:
    """Definition of a purchasable token package."""

    id: str
    name: str
    tokens: int
    price_usd: Decimal
    bonus_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.tokens + self.bonus_tokens


TOKEN_PACKAGES: dict[str, TokenPackage] = {
    p.id: p
    for p in [
        TokenPackage("starter", "Starter", 1_000, Decimal("9.99")),
        TokenPackage("basic", "Basic", 5_000, Decimal("39.99")),
        TokenPackage("pro", "Pro", 15_000, Decimal("99.99"), bonus_tokens=1_500),
        TokenPackage("enterprise", "Enterprise", 50_000, Decimal("299.99"), bonus_tokens=10_000),
    ]
}


# Unsafe placeholder that ships as the default – reject it at startup
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

    # --- Cloudflare Tunnel ---
    cf_api_token: str = Field(
        default="",
        description="Cloudflare API token with Tunnel:Edit permission",
    )
    cf_account_id: str = Field(
        default="",
        description="Cloudflare account ID",
    )
    cf_zone_id: str = Field(
        default="",
        description="Cloudflare zone ID for the tunnel domain (cloudin.space)",
    )
    cf_tunnel_domain: str = Field(
        default="gpu.cloudin.space",
        description="Base domain for GPU session tunnel hostnames",
    )

    # --- GPU Provisioning Worker ---
    gpu_provision_poll_interval_seconds: int = Field(
        default=15,
        ge=5,
        le=60,
        description="How often to poll pending/provisioning/resuming sessions",
    )
    gpu_provision_timeout_minutes: int = Field(
        default=20,
        ge=5,
        le=60,
        description="Max time per provisioning attempt before retry/failure",
    )
    gpu_resume_timeout_minutes: int = Field(
        default=5,
        ge=1,
        le=15,
        description="Max time to wait for a paused session to resume",
    )
    max_node_provisioning_retries: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Max provisioning attempts before marking session as failed",
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
    gpu_provision_worker_concurrency: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Max sessions processed concurrently per provisioning worker sweep",
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
            "The start-time reservation is computed as MIN_BILLABLE_MINUTES × rate "
            "(see src.api.services.gpu_session.service._MIN_BILLABLE_MINUTES). "
            "Keeping reservation and rate coupled prevents settings drift from "
            "causing over-refunds between session start and finalize."
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

    # Generation defaults
    default_steps: int = Field(default=12, description="Default generation steps")
    max_steps: int = Field(default=20, description="Maximum generation steps")
    default_cfg: float = Field(default=1.1, description="Default CFG scale")
    default_sampler: str = Field(default="euler", description="Default sampler")
    default_scheduler: str = Field(default="beta", description="Default scheduler")

    # -------------------------------------------------------------------------
    # System settings
    # -------------------------------------------------------------------------

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API server host")  # noqa: S104
    api_port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # ASGI Granian settings
    asgi_workers: int = Field(
        default=1,
        ge=1,
        le=32,
        description="Number of worker processes for ASGI server. Increase in production based on CPU cores and expected load.",
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
    max_upload_size_mb: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Maximum upload file size in MB",
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

    # Per-product NowPayments keys (vex.pics only — crypto payments for consumers)
    nowpayments_api_key_vex: str | None = Field(
        default=None,
        description="NowPayments API key for vex.pics",
    )
    nowpayments_ipn_secret_vex: str | None = Field(
        default=None,
        description="NowPayments IPN HMAC secret for vex.pics",
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
    # Validators
    # -------------------------------------------------------------------------

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
    def cf_configured(self) -> bool:
        """Check if Cloudflare Tunnel is configured."""
        return bool(self.cf_api_token and self.cf_account_id and self.cf_zone_id)

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
