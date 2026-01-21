"""Application configuration using pydantic-settings."""

from functools import lru_cache

from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


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
        description="Secret key for JWT signing (use strong random key in production)",
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


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


def reset_settings() -> None:
    """Reset cached settings (useful for testing)."""
    get_settings.cache_clear()
