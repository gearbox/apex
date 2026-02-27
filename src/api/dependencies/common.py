"""Dependency injection providers for Litestar."""

from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

from litestar.di import Provide
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.auth import AuthService
from src.api.services.billing import BillingService
from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.email import EmailService, LogEmailService, ResendEmailService
from src.api.services.email_verification import EmailVerificationService
from src.api.services.grok import GrokClient
from src.api.services.grok.job_service import GrokJobService
from src.api.services.job_manager import JobManager
from src.api.services.organization import OrganizationService
from src.api.services.payment import PaymentService
from src.api.services.pricing import PricingService
from src.api.services.storage import R2StorageService, R2StorageSettings
from src.api.services.unified_jobs import UnifiedJobService
from src.api.services.user import UserService
from src.api.services.user_content import UserContentService
from src.api.services.workflow_service import WorkflowService
from src.core.config import Settings, get_settings
from src.db import DatabaseManager, init_db
from src.db.repositories import UserRepository

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Service container (single global, initialized at app startup)
# -----------------------------------------------------------------------------


@dataclass
class ServiceContainer:
    """Container for all singleton services.

    Centralizes service instances in a single object, avoiding
    multiple module-level globals and long `global` declarations.
    """

    comfyui_client: ComfyUIClient | None = None
    job_manager: JobManager | None = None
    workflow_service: WorkflowService | None = None
    r2_storage: R2StorageService | None = None
    db_manager: DatabaseManager | None = None
    jwt_service: JWTService | None = None
    password_service: PasswordService | None = None
    grok_job_service: GrokJobService | None = None
    email_service: EmailService | None = None
    email_verification_service: EmailVerificationService | None = None
    unified_job_service: UnifiedJobService | None = None


_services = ServiceContainer()


# -----------------------------------------------------------------------------
# ComfyUI dependencies
# -----------------------------------------------------------------------------


async def get_comfyui_client() -> ComfyUIClient:
    """Provide ComfyUI client instance.

    Returns:
        Singleton ComfyUI client.

    Raises:
        RuntimeError: If client not initialized.
    """
    if _services.comfyui_client is None:
        raise RuntimeError("ComfyUI client not initialized")
    return _services.comfyui_client


async def get_job_manager() -> JobManager:
    """Provide job manager instance.

    Returns:
        Singleton job manager.

    Raises:
        RuntimeError: If manager not initialized.
    """
    if _services.job_manager is None:
        raise RuntimeError("Job manager not initialized")
    return _services.job_manager


async def get_workflow_service() -> WorkflowService:
    """Provide workflow service instance.

    Returns:
        Singleton workflow service.

    Raises:
        RuntimeError: If service not initialized.
    """
    if _services.workflow_service is None:
        raise RuntimeError("Workflow service not initialized")
    return _services.workflow_service


# -----------------------------------------------------------------------------
# Storage dependencies
# -----------------------------------------------------------------------------


async def get_r2_storage() -> R2StorageService:
    """Provide R2 storage service instance.

    Returns:
        Singleton R2 storage service.

    Raises:
        RuntimeError: If storage not initialized.
    """
    if _services.r2_storage is None:
        raise RuntimeError("R2 storage not initialized")
    return _services.r2_storage


async def get_db_session() -> AsyncGenerator[AsyncSession]:
    """Provide database session for request scope.

    Yields:
        Database session that auto-commits on success.
    """
    if _services.db_manager is None:
        raise RuntimeError("Database not initialized")

    async with _services.db_manager.session() as session:
        yield session


async def get_user_content(
    r2_storage: R2StorageService,
    session: AsyncSession,
    settings: Settings,
) -> UserContentService:
    """Provide user content service.

    Creates a new instance per request with the injected session.

    Args:
        r2_storage: R2 storage service.
        session: Database session.
        settings: Application settings.

    Returns:
        Configured UserContentService.
    """
    return UserContentService(
        storage=r2_storage,
        session=session,
        retention_days=settings.retention_days,
    )


# -----------------------------------------------------------------------------
# Auth & User dependencies
# -----------------------------------------------------------------------------


def get_jwt_service() -> JWTService:
    """Provide JWT service singleton.

    Returns:
        JWTService instance.

    Raises:
        RuntimeError: If not initialized.
    """
    if _services.jwt_service is None:
        raise RuntimeError("JWT service not initialized")
    return _services.jwt_service


def get_password_service() -> PasswordService:
    """Provide password service singleton.

    Returns:
        PasswordService instance.

    Raises:
        RuntimeError: If not initialized.
    """
    if _services.password_service is None:
        raise RuntimeError("Password service not initialized")
    return _services.password_service


async def get_auth_service(session: AsyncSession) -> AuthService:
    """Provide auth service for request scope.

    Args:
        session: Database session.

    Returns:
        AuthService instance with email verification.
    """
    repository = UserRepository(session)
    return AuthService(
        repository=repository,
        jwt_service=get_jwt_service(),
        password_service=get_password_service(),
        session=session,
        email_verification_service=get_email_verification_service(),
    )


async def get_user_service(session: AsyncSession) -> UserService:
    """Provide user service for request scope.

    Args:
        session: Database session.

    Returns:
        UserService instance.
    """
    repository = UserRepository(session)
    return UserService(
        repository=repository,
        password_service=get_password_service(),
    )


def get_email_service() -> EmailService:
    """Provide email service singleton.

    Returns:
        Configured EmailService (Log or Resend depending on settings).

    Raises:
        RuntimeError: If not initialized.
    """
    if _services.email_service is None:  # type: ignore[attr-defined]
        raise RuntimeError("Email service not initialized")
    return _services.email_service  # type: ignore[attr-defined]


def get_email_verification_service() -> EmailVerificationService:
    """Provide email verification service singleton."""
    if _services.email_verification_service is None:  # type: ignore[attr-defined]
        raise RuntimeError("Email verification service not initialized")
    return _services.email_verification_service  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Grok dependency providers
# -----------------------------------------------------------------------------


async def get_grok_job_service() -> GrokJobService | None:
    """Provide Grok job service (None if not configured)."""
    return _services.grok_job_service


# -----------------------------------------------------------------------------
# Unified job dependency providers
# -----------------------------------------------------------------------------


def get_unified_job_service() -> UnifiedJobService:
    """Provide unified job service singleton."""
    if _services.unified_job_service is None:  # type: ignore[attr-defined]
        raise RuntimeError("Unified job service not initialized")
    return _services.unified_job_service  # type: ignore[attr-defined]


# -----------------------------------------------------------------------------
# Billing dependencies
# -----------------------------------------------------------------------------


def get_billing_service() -> BillingService:
    """Provide BillingService singleton."""
    return BillingService()


def get_pricing_service() -> PricingService:
    """Provide PricingService singleton."""
    return PricingService()


def get_organization_service() -> OrganizationService:
    """Provide OrganizationService singleton."""
    return OrganizationService()


def get_payment_service() -> PaymentService:
    """Provide PaymentService singleton."""
    return PaymentService(
        billing_service=get_billing_service(),
        settings=get_settings(),
    )


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------


def provide_settings() -> Settings:
    """Provide settings instance.

    Returns:
        Application settings.
    """
    return get_settings()


# -----------------------------------------------------------------------------
# Lifecycle management
# -----------------------------------------------------------------------------


async def init_services(settings: Settings, base_path: Path | None = None) -> JWTService:
    """Initialize all service singletons.

    Called during application startup.

    Args:
        settings: Application settings.
        base_path: Base path for workflow files.
    Returns:
        JWT service for storing in app state.
    """
    # Initialize database
    _services.db_manager = init_db(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.db_echo,
    )
    logger.info("Database connection pool initialized")

    # Initialize ComfyUI client
    _services.comfyui_client = ComfyUIClient(settings)
    await _services.comfyui_client.connect()
    logger.info(f"Connected to ComfyUI at {settings.comfyui_base_url}")

    # Initialize job manager with client
    _services.job_manager = JobManager(_services.comfyui_client)

    # Initialize workflow service
    _services.workflow_service = WorkflowService(base_path=base_path)

    # Initialize R2 storage (if configured)
    if settings.r2_configured:
        r2_settings = R2StorageSettings(
            account_id=settings.r2_account_id,
            access_key_id=settings.r2_access_key_id,
            secret_access_key=settings.r2_secret_access_key,
            bucket_name=settings.r2_bucket_name,
            public_url_base=settings.r2_public_url_base,
            retention_days=settings.retention_days,
        )
        _services.r2_storage = R2StorageService(r2_settings)
        logger.info(f"R2 storage initialized for bucket: {settings.r2_bucket_name}")
    else:
        logger.warning("R2 storage not configured - storage endpoints will be unavailable")

    # Initialize Grok provider (if configured)
    if settings.grok_configured and _services.r2_storage is not None:
        grok_client = GrokClient(settings)
        await grok_client.connect()

        _services.grok_job_service = GrokJobService(
            grok_client=grok_client,
            storage=_services.r2_storage,
            retention_days=settings.retention_days,
        )
        await _services.grok_job_service.connect()
        logger.info("Grok provider initialized")
    else:
        logger.warning("Grok provider not configured - XAI_API_KEY not set or R2 not available")

    # Initialize authentication services
    jwt_config = JWTConfig(
        secret_key=settings.jwt_secret_key,
        algorithm=settings.jwt_algorithm,
        access_token_expire_minutes=settings.jwt_access_token_expire_minutes,
        refresh_token_expire_days=settings.jwt_refresh_token_expire_days,
        issuer=settings.jwt_issuer,
    )
    _services.jwt_service = JWTService(jwt_config)
    _services.password_service = PasswordService()
    logger.info("Authentication services initialized")

    # Initialize email service
    if settings.resend_api_key:
        _services.email_service = ResendEmailService(
            api_key=settings.resend_api_key,
            from_address=settings.email_from_address,
            from_name=settings.email_from_name,
        )
        logger.info("Email service initialized (Resend)")
    else:
        _services.email_service = LogEmailService()
        logger.info("Email service initialized (Log — no RESEND_API_KEY set)")

    # Initialize email verification service
    _services.email_verification_service = EmailVerificationService(
        email_service=_services.email_service,
        app_url=settings.app_url,
    )

    # Initialize unified job service
    _services.unified_job_service = UnifiedJobService(
        storage=_services.r2_storage,
        grok_job_service=_services.grok_job_service,
    )
    logger.info("Unified job service initialized")

    return _services.jwt_service  # Return for storing in app.state


async def shutdown_services() -> None:
    """Cleanup service resources.

    Called during application shutdown.
    """
    global _services

    if _services.comfyui_client is not None:
        await _services.comfyui_client.close()
        logger.info("ComfyUI client closed")

    if _services.r2_storage is not None:
        await _services.r2_storage.close()
        logger.info("R2 storage closed")

    if _services.db_manager is not None:
        await _services.db_manager.close()
        logger.info("Database connections closed")

    if _services.grok_job_service is not None:
        await _services.grok_job_service.close()
        logger.info("Grok job service closed")

    # Reset container to clean state
    _services = ServiceContainer()


# Dependency providers for Litestar
dependencies = {
    # Authentication services
    "auth_service": Provide(get_auth_service),
    "user_service": Provide(get_user_service),
    # Core services
    "comfyui_client": Provide(get_comfyui_client),
    "job_manager": Provide(get_job_manager),
    "workflow_service": Provide(get_workflow_service),
    "settings": Provide(provide_settings, sync_to_thread=False),
    # Storage services
    "r2_storage": Provide(get_r2_storage),
    "session": Provide(get_db_session),
    "user_content": Provide(get_user_content),
    # Grok services
    "grok_job_service": Provide(get_grok_job_service),
    # Billing services
    "billing_service": Provide(get_billing_service, sync_to_thread=False),
    "pricing_service": Provide(get_pricing_service, sync_to_thread=False),
    "organization_service": Provide(get_organization_service, sync_to_thread=False),
    "payment_service": Provide(get_payment_service, sync_to_thread=False),
    # Email services
    "email_verification_service": Provide(get_email_verification_service, sync_to_thread=False),
    # Unified job service
    "unified_job_service": Provide(get_unified_job_service, sync_to_thread=False),
}
