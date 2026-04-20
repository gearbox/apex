"""Dependency injection providers for Litestar."""

from __future__ import annotations

from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog
from litestar import Request
from litestar.di import Provide
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.middleware.rate_limit import init_rate_limiter
from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.aisha_job_service import AishaJobService
from src.api.services.auth import AuthService
from src.api.services.billing import BillingService
from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.content_proxy import ContentProxyService
from src.api.services.email import EmailService, LogEmailService, ResendEmailService
from src.api.services.email_verification import EmailVerificationService
from src.api.services.event_bus import EventBus
from src.api.services.gallery import GalleryService
from src.api.services.generation.service import GenerationService
from src.api.services.gpu_session.cleanup_worker import OrphanedTunnelCleanupWorker
from src.api.services.gpu_session.provisioning_worker import GpuProvisioningWorker
from src.api.services.gpu_session.service import GpuSessionService
from src.api.services.grok import GrokClient
from src.api.services.grok.job_service import GrokJobService
from src.api.services.health.service import HealthService
from src.api.services.health.worker import HealthSnapshotWorker
from src.api.services.idempotency import IdempotencyService
from src.api.services.organization import OrganizationService
from src.api.services.payment import PaymentService
from src.api.services.pricing import PricingService
from src.api.services.sse_ticket import SSETicketService
from src.api.services.storage import R2StorageService, R2StorageSettings
from src.api.services.unified_jobs import UnifiedJobService
from src.api.services.user import UserService
from src.api.services.user_content import UserContentService
from src.api.services.workflow_service import WorkflowService
from src.core.config import Settings, get_settings
from src.core.product import ProductConfig
from src.db import DatabaseManager, init_db
from src.db.repositories import UserRepository
from src.workers.token_cleanup import TokenCleanupWorker

logger = structlog.get_logger(__name__)

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
    aisha_job_service: AishaJobService | None = None
    workflow_service: WorkflowService | None = None
    r2_storage: R2StorageService | None = None
    db_manager: DatabaseManager | None = None
    jwt_service: JWTService | None = None
    password_service: PasswordService | None = None
    grok_job_service: GrokJobService | None = None
    email_service: EmailService | None = None
    email_verification_service: EmailVerificationService | None = None
    unified_job_service: UnifiedJobService | None = None
    token_cleanup_worker: TokenCleanupWorker | None = None
    generation_service: GenerationService | None = None
    event_bus: EventBus | None = None
    sse_ticket_service: SSETicketService | None = None
    health_service: HealthService | None = None
    health_snapshot_worker: HealthSnapshotWorker | None = None
    health_http_client: httpx.AsyncClient | None = None
    # GPU session stack (optional — requires Vast.ai + CF configuration)
    gpu_session_service: GpuSessionService | None = None
    gpu_provisioning_worker: GpuProvisioningWorker | None = None
    orphaned_tunnel_cleanup_worker: OrphanedTunnelCleanupWorker | None = None
    gpu_session_http_client: httpx.AsyncClient | None = None


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
    product_id: str,
) -> UserContentService:
    """Provide user content service.

    Creates a new instance per request with the injected session.

    Args:
        r2_storage: R2 storage service.
        session: Database session.
        settings: Application settings.
        product_id: Product slug from ProductMiddleware.

    Returns:
        Configured UserContentService.
    """
    return UserContentService(
        storage=r2_storage,
        session=session,
        product_id=product_id,
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
        r2_storage=_services.r2_storage,
    )


def get_email_service() -> EmailService:
    """Provide email service singleton.

    Returns:
        Configured EmailService (Log or Resend depending on settings).

    Raises:
        RuntimeError: If not initialized.
    """
    if _services.email_service is None:
        raise RuntimeError("Email service not initialized")
    return _services.email_service


def get_email_verification_service() -> EmailVerificationService:
    """Provide email verification service singleton."""
    if _services.email_verification_service is None:
        raise RuntimeError("Email verification service not initialized")
    return _services.email_verification_service


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
    if _services.unified_job_service is None:
        raise RuntimeError("Unified job service not initialized")
    return _services.unified_job_service


def get_generation_service() -> GenerationService:
    """Provide GenerationService singleton."""
    if _services.generation_service is None:
        raise RuntimeError("GenerationService not initialized")
    return _services.generation_service


# -----------------------------------------------------------------------------
# Billing dependencies
# -----------------------------------------------------------------------------


def get_event_bus() -> EventBus:
    """Provide EventBus singleton."""
    if _services.event_bus is None:
        raise RuntimeError("EventBus not initialized")
    return _services.event_bus


def get_sse_ticket_service() -> SSETicketService:
    """Provide SSETicketService singleton."""
    if _services.sse_ticket_service is None:
        raise RuntimeError("SSETicketService not initialized")
    return _services.sse_ticket_service


def get_billing_service() -> BillingService:
    """Provide BillingService singleton."""
    return BillingService(event_bus=_services.event_bus)


def get_idempotency_service() -> IdempotencyService:
    """Provide IdempotencyService instance."""
    settings = get_settings()
    return IdempotencyService(ttl_hours=settings.idempotency_key_ttl_hours)


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


def get_health_service() -> HealthService:
    """Provide HealthService singleton."""
    if _services.health_service is None:
        raise RuntimeError("HealthService not initialized")
    return _services.health_service


async def get_gpu_session_service() -> GpuSessionService:
    """Provide GpuSessionService singleton (503 if GPU stack not configured)."""
    if _services.gpu_session_service is None:
        from litestar.exceptions import ServiceUnavailableException

        raise ServiceUnavailableException(
            detail="GPU session service not available (Vast.ai/CF not configured)"
        )
    return _services.gpu_session_service


def get_content_proxy(
    r2_storage: R2StorageService,
    settings: Settings,
) -> ContentProxyService:
    """Provide ContentProxyService per-request."""
    return ContentProxyService(storage=r2_storage, settings=settings)


def get_gallery_service(session: AsyncSession) -> GalleryService:
    """Provide GalleryService per-request."""
    return GalleryService(session=session)


def provide_settings() -> Settings:
    """Provide settings instance.

    Returns:
        Application settings.
    """
    return get_settings()


# -----------------------------------------------------------------------------
# Product dependencies
# -----------------------------------------------------------------------------


def get_product_config(request: Request) -> ProductConfig:  # type: ignore[type-arg]
    """Extract product config set by ProductMiddleware.

    Args:
        request: Litestar Request object.

    Returns:
        ProductConfig resolved by the middleware.

    Raises:
        ValueError: If ProductMiddleware did not run.
    """
    config: ProductConfig | None = getattr(request.state, "product_config", None)
    if config is None:
        raise ValueError(
            "ProductMiddleware did not run — product_config missing from request state"
        )
    return config


def get_product_id(request: Request) -> str:  # type: ignore[type-arg]
    """Extract product slug set by ProductMiddleware.

    Args:
        request: Litestar Request object.

    Returns:
        Product slug string.
    """
    return request.state.product_id  # type: ignore[no-any-return]


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
    logger.info("db.pool_initialized")

    # Initialize Redis (required for pub/sub and rate limiting)
    from src.core.redis import init_redis_pool

    if settings.redis_url:
        init_redis_pool(settings.redis_url)
        _services.event_bus = EventBus()
        logger.info("event_bus.initialized")
        _services.sse_ticket_service = SSETicketService(ttl_seconds=settings.sse_ticket_ttl_seconds)
        logger.info("sse_ticket_service.initialized")
    else:
        logger.warning("redis.not_configured — event_bus and sse_ticket_service disabled")

    # Initialize rate limiter
    init_rate_limiter(settings)

    # Initialize ComfyUI client
    _services.comfyui_client = ComfyUIClient(settings.comfyui_base_url)
    await _services.comfyui_client.connect()
    logger.info("comfyui.connected", url=settings.comfyui_base_url)

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
        logger.info("r2.initialized", bucket=settings.r2_bucket_name)
    else:
        logger.warning("r2.not_configured")

    # Initialize Aisha job service (if R2 is configured)
    if _services.r2_storage is not None and _services.comfyui_client is not None:
        _services.aisha_job_service = AishaJobService(
            comfyui_client=_services.comfyui_client,
            storage=_services.r2_storage,
            retention_days=settings.retention_days,
        )
        logger.info("aisha_job_service.initialized")

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
        logger.info("grok.initialized")
    else:
        logger.warning("grok.not_configured")

    # Start Grok video polling worker
    if settings.grok_configured and _services.grok_job_service is not None:
        if settings.grok_video_worker_in_process:
            from src.api.services.grok.video_worker import GrokVideoWorkerManager

            await GrokVideoWorkerManager.start(
                db_manager=_services.db_manager,
                job_service=_services.grok_job_service,
                settings=settings,
                event_bus=_services.event_bus,
            )
            logger.info("grok.video_worker_in_process_started")
        else:
            logger.info(
                "grok.video_worker_external_required",
                hint="Run 'python -m src.workers.grok_video' as a separate process",
            )

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
    logger.info("auth.initialized")

    # Initialize email service
    if settings.resend_api_key:
        _services.email_service = ResendEmailService(
            api_key=settings.resend_api_key,
            from_address=settings.email_from_address,
            from_name=settings.email_from_name,
        )
        logger.info("email.initialized", provider="resend")
    else:
        _services.email_service = LogEmailService()
        logger.info("email.initialized", provider="log")

    # Initialize email verification service
    _services.email_verification_service = EmailVerificationService(
        email_service=_services.email_service,
        app_url=settings.app_url,
        app_name=settings.app_name,
    )

    # Initialize unified job service
    _services.unified_job_service = UnifiedJobService(
        storage=_services.r2_storage,
        grok_job_service=_services.grok_job_service,
        aisha_job_service=_services.aisha_job_service,
    )
    logger.info("unified_job_service.initialized")

    # Initialize GPU session stack (requires both Vast.ai + CF to be configured)
    if settings.vastai_configured and settings.cf_configured:
        from src.api.services.bundle_index import BundleIndexService
        from src.api.services.cloudflare.client import CloudflareTunnelClient
        from src.api.services.vastai.client import VastAIClient

        _services.gpu_session_http_client = httpx.AsyncClient(timeout=30.0)

        vastai_client = VastAIClient(
            http_client=_services.gpu_session_http_client,
            api_key=settings.vastai_api_key,
        )
        cf_client = CloudflareTunnelClient(
            http_client=_services.gpu_session_http_client,
            api_token=settings.cf_api_token,
            account_id=settings.cf_account_id,
            zone_id=settings.cf_zone_id,
            tunnel_domain=settings.cf_tunnel_domain,
        )
        bundle_index = BundleIndexService(
            repo_url=settings.ai_bundles_repo_url,
            github_token=settings.ai_bundles_github_token,
            sync_interval_minutes=settings.ai_bundles_sync_interval_minutes,
        )
        await bundle_index.start()

        billing_service_for_worker = BillingService(event_bus=_services.event_bus)

        _services.gpu_session_service = GpuSessionService(
            vastai_client=vastai_client,
            cf_client=cf_client,
            bundle_index=bundle_index,
            session_factory=_services.db_manager.session_factory,
            settings=settings,
            event_bus=_services.event_bus,
        )

        _services.gpu_provisioning_worker = GpuProvisioningWorker(
            session_factory=_services.db_manager.session_factory,
            vastai_client=vastai_client,
            cf_client=cf_client,
            bundle_index=bundle_index,
            http_client=_services.gpu_session_http_client,
            settings=settings,
            billing_service=billing_service_for_worker,
            event_bus=_services.event_bus,
        )
        await _services.gpu_provisioning_worker.start()

        _services.orphaned_tunnel_cleanup_worker = OrphanedTunnelCleanupWorker(
            session_factory=_services.db_manager.session_factory,
            cf_client=cf_client,
            settings=settings,
        )
        await _services.orphaned_tunnel_cleanup_worker.start()

        logger.info("gpu_session_stack.initialized")
    else:
        logger.warning(
            "gpu_session_stack.not_configured",
            hint="Set VASTAI_API_KEY and CF_API_TOKEN/CF_ACCOUNT_ID/CF_ZONE_ID to enable GPU sessions",
        )

    # Initialize unified generation service
    from src.api.services.generation.aisha_provider import AishaGenerationProvider
    from src.api.services.generation.grok_provider import GrokGenerationProvider
    from src.core.enums import Provider

    generation_providers: dict[Provider, object] = {
        Provider.AISHA: AishaGenerationProvider(
            workflow_service=_services.workflow_service,
            gpu_session_service=_services.gpu_session_service,
        )
    }

    if _services.grok_job_service is not None and _services.r2_storage is not None:
        generation_providers[Provider.GROK] = GrokGenerationProvider(
            grok_job_service=_services.grok_job_service,
            r2_storage=_services.r2_storage,
        )

    from src.api.services.generation.rate_limiter import ModelRateLimiter

    model_rate_limiter = ModelRateLimiter()

    _services.generation_service = GenerationService(
        providers=generation_providers,  # type: ignore[arg-type]
        billing_service=get_billing_service(),
        pricing_service=get_pricing_service(),
        rate_limiter=model_rate_limiter,
        event_bus=_services.event_bus,
    )
    logger.info(
        "generation_service.initialized",
        providers=[p.value for p in generation_providers],
    )

    # Initialize and start token cleanup worker
    _services.token_cleanup_worker = TokenCleanupWorker(
        db_manager=_services.db_manager,
        interval=settings.token_cleanup_interval_seconds,
    )
    await _services.token_cleanup_worker.start()
    logger.info("token_cleanup_worker.started")

    # Initialize health check registry and service
    from src.api.services.health.checkers.cloud_provider import GrokChecker
    from src.api.services.health.checkers.gpu_session import GpuSessionReconciler
    from src.api.services.health.checkers.infrastructure import (
        PostgresChecker,
        R2Checker,
        RedisChecker,
    )
    from src.api.services.health.checkers.platform_api import VastAIChecker
    from src.api.services.health.registry import HealthCheckRegistry

    # Shared HTTP client for health check probes (cloud providers, platform APIs)
    health_http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, connect=5.0),
        follow_redirects=True,
    )
    _services.health_http_client = health_http_client

    health_registry = HealthCheckRegistry()
    health_registry.register(PostgresChecker(session_factory=_services.db_manager.session_factory))
    if settings.redis_url:
        from src.core.redis import get_redis_client

        health_registry.register(RedisChecker(redis=get_redis_client()))
    if _services.r2_storage is not None:
        health_registry.register(R2Checker(r2_storage=_services.r2_storage))

    # Cloud provider checkers (per-product)
    if settings.grok_configured:
        for product_slug in ["vex", "synthara"]:
            health_registry.register(
                GrokChecker(
                    http_client=health_http_client,
                    api_base_url=settings.xai_api_base_url,
                    api_key=settings.xai_api_key,
                    product_id=product_slug,
                )
            )

    # Platform API checkers (global)
    health_registry.register(
        VastAIChecker(
            http_client=health_http_client,
            api_key=settings.vastai_api_key if settings.vastai_configured else None,
        )
    )

    # GPU session reconciler (global — reconciles across all products)
    health_registry.register(
        GpuSessionReconciler(
            session_factory=_services.db_manager.session_factory,
            http_client=health_http_client,
        )
    )

    _services.health_service = HealthService(
        registry=health_registry,
        timeout_seconds=settings.health_check_timeout_seconds,
    )
    logger.info("health_service.initialized")

    # Initialize and start health snapshot worker
    _services.health_snapshot_worker = HealthSnapshotWorker(
        health_service=_services.health_service,
        db_manager=_services.db_manager,
        interval_seconds=settings.health_snapshot_interval_seconds,
        retention_days=settings.health_snapshot_retention_days,
        redis_url=settings.redis_url,
    )
    await _services.health_snapshot_worker.start()

    return _services.jwt_service  # Return for storing in app.state


async def shutdown_services() -> None:
    """Cleanup service resources.

    Called during application shutdown.
    """
    global _services

    # Stop GPU session workers before closing their dependencies
    if _services.gpu_provisioning_worker is not None:
        await _services.gpu_provisioning_worker.stop()

    if _services.orphaned_tunnel_cleanup_worker is not None:
        await _services.orphaned_tunnel_cleanup_worker.stop()

    if _services.gpu_session_http_client is not None:
        await _services.gpu_session_http_client.aclose()
        logger.info("gpu_session_http_client.closed")

    if _services.comfyui_client is not None:
        await _services.comfyui_client.close()
        logger.info("comfyui.closed")

    if _services.r2_storage is not None:
        await _services.r2_storage.close()
        logger.info("r2.closed")

    if _services.db_manager is not None:
        await _services.db_manager.close()
        logger.info("db.closed")

    if _services.grok_job_service is not None:
        await _services.grok_job_service.close()
        logger.info("grok.closed")

    if _services.token_cleanup_worker is not None:
        await _services.token_cleanup_worker.stop()

    settings = get_settings()
    if settings.grok_configured and settings.grok_video_worker_in_process:
        from src.api.services.grok.video_worker import GrokVideoWorkerManager

        await GrokVideoWorkerManager.stop()

    if _services.health_snapshot_worker is not None:
        await _services.health_snapshot_worker.stop()

    if _services.health_http_client is not None:
        await _services.health_http_client.aclose()
        logger.info("health_http_client.closed")

    from src.core.redis import close_redis_pool

    await close_redis_pool()

    # Reset container to clean state
    _services = ServiceContainer()


# Dependency providers for Litestar
dependencies = {
    # Authentication services
    "auth_service": Provide(get_auth_service),
    "user_service": Provide(get_user_service),
    # Core services
    "comfyui_client": Provide(get_comfyui_client),
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
    # Unified generation service
    "generation_service": Provide(get_generation_service, sync_to_thread=False),
    # Real-time events
    "event_bus": Provide(get_event_bus, sync_to_thread=False),
    "sse_ticket_service": Provide(get_sse_ticket_service, sync_to_thread=False),
    # Product context (resolved by ProductMiddleware)
    "product_config": Provide(get_product_config, sync_to_thread=False),
    "product_id": Provide(get_product_id, sync_to_thread=False),
    # Content proxy
    "content_proxy": Provide(get_content_proxy, sync_to_thread=False),
    # Gallery
    "gallery_service": Provide(get_gallery_service, sync_to_thread=False),
    # Idempotency
    "idempotency_service": Provide(get_idempotency_service, sync_to_thread=False),
    # Health
    "health_service": Provide(get_health_service, sync_to_thread=False),
    # GPU sessions
    "gpu_session_service": Provide(get_gpu_session_service),
}
