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

    # API settings
    api_host: str = Field(default="0.0.0.0", description="API server host")
    api_port: int = Field(default=8000, description="API server port")
    debug: bool = Field(default=False, description="Enable debug mode")

    # Generation defaults
    default_steps: int = Field(default=12, description="Default generation steps")
    max_steps: int = Field(default=20, description="Maximum generation steps")
    default_cfg: float = Field(default=1.1, description="Default CFG scale")
    default_sampler: str = Field(default="euler", description="Default sampler")
    default_scheduler: str = Field(default="beta", description="Default scheduler")

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

    # Billing & Payment settings
    grok_moderation_billing_policy: Literal["charge", "refund"] = Field(
        default="charge",
        description="Policy for Grok moderation: 'charge' or 'refund'",
    )

    # Stripe
    stripe_secret_key: str = Field(
        default="",
        description="Stripe secret API key",
    )
    stripe_webhook_secret: str = Field(
        default="",
        description="Stripe webhook endpoint signing secret",
    )

    # NowPayments
    nowpayments_api_key: str = Field(
        default="",
        description="NowPayments API key",
    )
    nowpayments_ipn_secret: str = Field(
        default="",
        description="NowPayments IPN HMAC secret",
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

    @computed_field
    @property
    def comfyui_base_url(self) -> str:
        """Construct ComfyUI base URL."""
        return f"http://{self.comfyui_host}:{self.comfyui_port}"

    @computed_field
    @property
    def r2_endpoint_url(self) -> str:
        """Construct R2 endpoint URL."""
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"

    @computed_field
    @property
    def max_upload_size_bytes(self) -> int:
        """Maximum upload size in bytes."""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def r2_configured(self) -> bool:
        """Check if R2 is properly configured."""
        return bool(self.r2_account_id and self.r2_access_key_id and self.r2_secret_access_key)

    @computed_field
    @property
    def grok_configured(self) -> bool:
        """Check if Grok API is configured."""
        return bool(self.xai_api_key)

    @property
    def grok_billing(self) -> GrokBillingConfig:
        """Get Grok billing configuration."""
        return GrokBillingConfig(
            moderation_billing_policy=self.grok_moderation_billing_policy,
        )

    @property
    def stripe_configured(self) -> bool:
        """Check if Stripe is configured."""
        return bool(self.stripe_secret_key and self.stripe_webhook_secret)

    @property
    def nowpayments_configured(self) -> bool:
        """Check if NowPayments is configured."""
        return bool(self.nowpayments_api_key and self.nowpayments_ipn_secret)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reset_settings() -> None:
    """Reset cached settings (useful for testing)."""
    get_settings.cache_clear()
