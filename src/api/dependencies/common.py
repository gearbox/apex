"""Dependency injection providers for Litestar."""

from __future__ import annotations

from collections.abc import AsyncGenerator  # noqa: TC003
from dataclasses import dataclass
from typing import TYPE_CHECKING

import httpx
import structlog
from litestar import Request  # noqa: TC002
from litestar.di import Provide
from sqlalchemy.ext.asyncio import AsyncSession  # noqa: TC002

from src.api.middleware.rate_limit import init_rate_limiter
from src.api.security import JWTConfig, JWTService, PasswordService
from src.api.services.admin_notifications import AdminNotificationService
from src.api.services.age_verification import AgeVerificationService
from src.api.services.auth import AuthService
from src.api.services.billing import BillingService
from src.api.services.bundle_index import BundleIndexService
from src.api.services.content_proxy import ContentProxyService
from src.api.services.content_retention import ContentRetentionService
from src.api.services.email import EmailService, LogEmailService, ResendEmailService
from src.api.services.email_verification import EmailVerificationService
from src.api.services.event_bus import EventBus
from src.api.services.frames.service import FrameExtractionService
from src.api.services.frames.worker import FrameExtractionWorker
from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.generation.service import GenerationService
from src.api.services.gpu_session.billing_reconciler_worker import BillingReconcilerWorker
from src.api.services.gpu_session.cleanup_worker import OrphanedTunnelCleanupWorker
from src.api.services.gpu_session.credit_guard import SessionCreditGuard
from src.api.services.gpu_session.node_cooldown import (
    NodeCooldownStore,
    NullNodeCooldownStore,
    RedisNodeCooldownStore,
)
from src.api.services.gpu_session.operation_event_service import OperationEventService
from src.api.services.gpu_session.provisioning_worker import GpuProvisioningWorker
from src.api.services.gpu_session.service import GpuSessionService
from src.api.services.grok import GrokClient
from src.api.services.grok.job_service import GrokJobService
from src.api.services.grok.video_worker import GrokVideoWorker
from src.api.services.health.checkers.workers import WorkerHeartbeatChecker
from src.api.services.health.service import HealthService
from src.api.services.health.worker import HealthSnapshotCleanupWorker, HealthSnapshotWorker
from src.api.services.idempotency import IdempotencyService
from src.api.services.jobs.sweep import JobSweepService
from src.api.services.library import LibraryService
from src.api.services.library_project import LibraryProjectService
from src.api.services.library_tag import LibraryTagService
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.organization import OrganizationService
from src.api.services.payment_currency_logos import LOGO_KEY_PREFIX, LogoCacheService
from src.api.services.payment_currency_sync import PaymentCurrencySyncService
from src.api.services.payment_provider_state import PaymentProviderStateService
from src.api.services.payments import GatewayRegistry, PaymentService
from src.api.services.payments.nowpayments_gateway import NowPaymentsGateway
from src.api.services.payments.stripe_gateway import StripeGateway
from src.api.services.pricing import PricingService
from src.api.services.push import PushService, PywebpushSender
from src.api.services.sse_ticket import SSETicketService
from src.api.services.storage import R2StorageService, R2StorageSettings, StorageError
from src.api.services.telegram.sender import HttpxTelegramSender
from src.api.services.token_revocation import TokenRevocationService
from src.api.services.unified_jobs import UnifiedJobService
from src.api.services.user import UserService
from src.api.services.user_content import UserContentService
from src.api.services.workflow import WorkflowService
from src.core.config import Settings, get_settings
from src.core.enums import WorkerMode
from src.core.product import ProductConfig  # noqa: TC001
from src.db import DatabaseManager, init_db
from src.db.repositories import UserRepository
from src.workers.aisha_job_poller import AishaJobPoller, AishaPollerConfig
from src.workers.content_retention import ContentRetentionWorker
from src.workers.payment_currency_sync import PaymentCurrencySyncWorker
from src.workers.push_dispatcher import PushDispatcher
from src.workers.telegram_dispatcher import TelegramDispatcher
from src.workers.telegram_link_poller import TelegramLinkPoller
from src.workers.token_cleanup import TokenCleanupWorker

if TYPE_CHECKING:
    from src.api.services.generation.base import GenerationProvider
    from src.workers.base import PeriodicWorker

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

    aisha_job_poller: AishaJobPoller | None = None
    workflow_service: WorkflowService | None = None
    r2_storage: R2StorageService | None = None
    r2_public_assets_storage: R2StorageService | None = None
    db_manager: DatabaseManager | None = None
    jwt_service: JWTService | None = None
    token_revocation_service: TokenRevocationService | None = None
    password_service: PasswordService | None = None
    billing_service: BillingService | None = None
    payment_provider_state_service: PaymentProviderStateService | None = None
    payment_service: PaymentService | None = None
    payment_currency_sync_service: PaymentCurrencySyncService | None = None
    payment_currency_sync_worker: PaymentCurrencySyncWorker | None = None
    grok_job_service: GrokJobService | None = None
    grok_video_worker: GrokVideoWorker | None = None
    email_service: EmailService | None = None
    email_verification_service: EmailVerificationService | None = None
    unified_job_service: UnifiedJobService | None = None
    token_cleanup_worker: TokenCleanupWorker | None = None
    content_retention_worker: ContentRetentionWorker | None = None
    frame_extraction_worker: FrameExtractionWorker | None = None
    generation_service: GenerationService | None = None
    event_bus: EventBus | None = None
    sse_ticket_service: SSETicketService | None = None
    health_service: HealthService | None = None
    health_snapshot_worker: HealthSnapshotWorker | None = None
    health_snapshot_cleanup_worker: HealthSnapshotCleanupWorker | None = None
    health_http_client: httpx.AsyncClient | None = None
    # GPU session stack (optional — requires Vast.ai + CF configuration)
    gpu_session_service: GpuSessionService | None = None
    gpu_provisioning_worker: GpuProvisioningWorker | None = None
    orphaned_tunnel_cleanup_worker: OrphanedTunnelCleanupWorker | None = None
    billing_reconciler_worker: BillingReconcilerWorker | None = None
    gpu_session_http_client: httpx.AsyncClient | None = None
    bundle_index: BundleIndexService | None = None
    # Operation receiver — initialized whenever the DB is available (not GPU-stack-gated)
    operation_event_service: OperationEventService | None = None
    # Web Push (optional — requires VAPID keys + Redis)
    push_service: PushService | None = None
    push_dispatcher: PushDispatcher | None = None
    # Ops events (admin notifications) — always constructed; enabled=False no-ops without Redis
    ops_event_bus: OpsEventBus | None = None
    # Telegram ops notifications (optional — requires bot token + Redis)
    telegram_sender: HttpxTelegramSender | None = None
    telegram_dispatcher: TelegramDispatcher | None = None
    telegram_link_poller: TelegramLinkPoller | None = None
    admin_notification_service: AdminNotificationService | None = None


_services = ServiceContainer()


# -----------------------------------------------------------------------------
# ComfyUI dependencies
# -----------------------------------------------------------------------------


def get_workflow_service() -> WorkflowService:
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


def get_r2_storage() -> R2StorageService:
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


def get_user_content(
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
        max_input_megapixels=settings.image_max_input_megapixels,
        video_max_seconds=settings.frame_extract_max_video_seconds,
        ffmpeg_timeout_seconds=settings.frame_extract_ffmpeg_timeout_seconds,
    )


def get_frame_extraction_service(
    r2_storage: R2StorageService,
    session: AsyncSession,
    settings: Settings,
    product_id: str,
) -> FrameExtractionService:
    """Provide frame extraction service.

    Creates a new instance per request with the injected session.
    """
    return FrameExtractionService(
        session=session,
        r2_storage=r2_storage,
        product_id=product_id,
        preview_url_ttl_seconds=settings.frame_preview_url_ttl_seconds,
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


def get_token_revocation_service() -> TokenRevocationService:
    """Provide TokenRevocationService singleton.

    Returns:
        TokenRevocationService instance (a working no-op if redis_url is
        unset — see the service docstring for the fail-open posture).

    Raises:
        RuntimeError: If not initialized.
    """
    if _services.token_revocation_service is None:
        raise RuntimeError("TokenRevocationService not initialized")
    return _services.token_revocation_service


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


def get_auth_service(session: AsyncSession) -> AuthService:
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
        ops_event_bus=get_ops_event_bus(),
        token_revocation_service=get_token_revocation_service(),
    )


def get_user_service(session: AsyncSession) -> UserService:
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
        age_verification_service=AgeVerificationService(),
        r2_storage=_services.r2_storage,
        token_revocation_service=get_token_revocation_service(),
        ops_event_bus=get_ops_event_bus(),
        session=session,
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


def get_grok_job_service() -> GrokJobService | None:
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


def get_ops_event_bus() -> OpsEventBus:
    """Provide OpsEventBus singleton."""
    if _services.ops_event_bus is None:
        raise RuntimeError("OpsEventBus not initialized")
    return _services.ops_event_bus


def get_sse_ticket_service() -> SSETicketService:
    """Provide SSETicketService singleton."""
    if _services.sse_ticket_service is None:
        raise RuntimeError("SSETicketService not initialized")
    return _services.sse_ticket_service


def get_billing_service() -> BillingService:
    """Provide the process-wide BillingService singleton.

    Raises:
        RuntimeError: If not initialized.
    """
    if _services.billing_service is None:
        raise RuntimeError("BillingService not initialized")
    return _services.billing_service


def get_idempotency_service() -> IdempotencyService:
    """Provide IdempotencyService instance."""
    settings = get_settings()
    return IdempotencyService(
        ttl_hours=settings.idempotency_key_ttl_hours,
        processing_stale_after_seconds=settings.idempotency_processing_stale_seconds,
    )


def get_pricing_service() -> PricingService:
    """Provide PricingService singleton."""
    return PricingService()


def get_organization_service() -> OrganizationService:
    """Provide OrganizationService singleton."""
    return OrganizationService()


def get_payment_service() -> PaymentService:
    """Provide PaymentService singleton."""
    if _services.payment_service is None:
        raise RuntimeError("PaymentService not initialized")
    return _services.payment_service


def get_payment_provider_state_service() -> PaymentProviderStateService:
    """Provide the runtime payment provider state service singleton."""
    if _services.payment_provider_state_service is None:
        raise RuntimeError("PaymentProviderStateService not initialized")
    return _services.payment_provider_state_service


def get_payment_currency_sync_service() -> PaymentCurrencySyncService:
    """Provide the payment currency catalog sync service singleton.

    Raises:
        RuntimeError: If not initialized (requires R2 to be configured).
    """
    if _services.payment_currency_sync_service is None:
        raise RuntimeError(
            "PaymentCurrencySyncService not initialized (requires R2 to be configured)"
        )
    return _services.payment_currency_sync_service


def build_gateway_registry(settings: Settings) -> GatewayRegistry:
    """Build the default complete payment gateway registry."""
    return GatewayRegistry([StripeGateway(settings), NowPaymentsGateway(settings)])


# -----------------------------------------------------------------------------
# Settings
# -----------------------------------------------------------------------------


def get_health_service() -> HealthService:
    """Provide HealthService singleton."""
    if _services.health_service is None:
        raise RuntimeError("HealthService not initialized")
    return _services.health_service


def get_gpu_session_service() -> GpuSessionService:
    """Provide GpuSessionService singleton (503 if GPU stack not configured)."""
    if _services.gpu_session_service is None:
        from litestar.exceptions import ServiceUnavailableException

        raise ServiceUnavailableException(
            detail="GPU session service not available (Vast.ai/CF not configured)"
        )
    return _services.gpu_session_service


def get_operation_event_service() -> OperationEventService:
    """Provide OperationEventService singleton (503 if DB not initialized)."""
    if _services.operation_event_service is None:
        from litestar.exceptions import ServiceUnavailableException

        raise ServiceUnavailableException(detail="Operation event service not available")
    return _services.operation_event_service


def get_push_service() -> PushService:
    """Provide PushService singleton (503 if push is not configured)."""
    if _services.push_service is None:
        from litestar.exceptions import ServiceUnavailableException

        raise ServiceUnavailableException(
            detail="Push notifications not available (VAPID keys/Redis not configured)"
        )
    return _services.push_service


def get_admin_notification_service() -> AdminNotificationService:
    """Provide the process-wide AdminNotificationService singleton.

    Single instance for the whole process (like BillingService) because it
    lazily caches the resolved Telegram bot username across calls — a
    per-request instance would re-fetch getMe() on every request.
    """
    if _services.admin_notification_service is None:
        raise RuntimeError("AdminNotificationService not initialized")
    return _services.admin_notification_service


def get_content_proxy(
    r2_storage: R2StorageService,
    settings: Settings,
) -> ContentProxyService:
    """Provide ContentProxyService per-request."""
    return ContentProxyService(storage=r2_storage, settings=settings)


def get_library_service(session: AsyncSession) -> LibraryService:
    """Provide LibraryService per-request."""
    return LibraryService(session=session)


def get_library_project_service(session: AsyncSession) -> LibraryProjectService:
    """Provide LibraryProjectService per-request."""
    return LibraryProjectService(session=session)


def get_library_tag_service(session: AsyncSession) -> LibraryTagService:
    """Provide LibraryTagService per-request."""
    return LibraryTagService(session=session)


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


async def init_services(settings: Settings) -> JWTService:
    """Initialize all service singletons.

    Called during application startup.

    Returns:
        JWT service for storing in app state.
    """
    # Whether this process starts periodic background worker loops.
    # api_only Granian processes construct every request-time service as
    # usual but skip every worker's construction and start() call.
    workers_enabled = settings.worker_mode is WorkerMode.all
    redis_enabled = settings.redis_url is not None
    if not workers_enabled:
        logger.info("workers.disabled_by_mode", worker_mode=settings.worker_mode.value)

    # Initialize database
    _services.db_manager = init_db(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.db_echo,
    )
    logger.info("db.pool_initialized")

    # Initialize operation event service (only needs the DB session factory).
    _services.operation_event_service = OperationEventService(
        session_factory=_services.db_manager.session_factory
    )
    logger.info("operation_event_service.initialized")

    # Initialize Redis (required for pub/sub and rate limiting). Three pools —
    # see src/core/redis.py module docstring: the shared pool serves
    # short-lived auth-path operations (token revocation, SSE tickets), the
    # operational pool serves health checks and leases, and the SSE pool
    # serves EventBus.subscribe's per-client long-lived connections.
    from src.core.redis import (
        get_operational_redis_client,
        get_redis_client,
        init_operational_redis_pool,
        init_redis_pool,
        init_sse_redis_pool,
    )

    if settings.redis_url:
        init_redis_pool(
            settings.redis_url,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            health_check_interval=settings.redis_health_check_interval_seconds,
            max_connections=settings.redis_max_connections,
        )
        init_sse_redis_pool(
            settings.redis_url,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            health_check_interval=settings.redis_health_check_interval_seconds,
            max_connections=settings.redis_sse_max_connections,
        )
        init_operational_redis_pool(
            settings.redis_url,
            socket_connect_timeout=settings.redis_operational_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_operational_socket_timeout_seconds,
            health_check_interval=settings.redis_health_check_interval_seconds,
            max_connections=settings.redis_operational_max_connections,
        )
        _services.sse_ticket_service = SSETicketService(ttl_seconds=settings.sse_ticket_ttl_seconds)
        logger.info("sse_ticket_service.initialized")
    else:
        logger.warning(
            "redis.not_configured — event bus publishes disabled, sse_ticket_service unavailable"
        )
    _services.event_bus = EventBus(enabled=settings.redis_url is not None)
    logger.info("event_bus.initialized", enabled=settings.redis_url is not None)

    _services.ops_event_bus = OpsEventBus(enabled=settings.redis_url is not None)
    logger.info("ops_event_bus.initialized", enabled=settings.redis_url is not None)

    # Token revocation (issue #142) — a working no-op when Redis isn't
    # configured, so call sites never branch on availability. TTL for the
    # bulk-revocation epoch key derives from the *upper bound* of the config
    # fields it protects (F3), not their current values — see
    # Settings.max_revocation_epoch_ttl_seconds.
    _services.token_revocation_service = TokenRevocationService(
        get_redis_client() if settings.redis_url else None,
        max_token_ttl_seconds=settings.max_revocation_epoch_ttl_seconds,
        access_token_lifetime_seconds=settings.jwt_access_token_expire_minutes * 60,
    )
    logger.info("token_revocation_service.initialized", enabled=settings.redis_url is not None)

    # Single BillingService singleton for the whole process — it is stateless
    # (a pure event-builder; publishing is the caller's job via
    # EventBus.publish_balance), so one instance safely serves every caller.
    _services.billing_service = BillingService()
    logger.info("billing_service.initialized")

    # Built once and shared with PaymentCurrencySyncService below — gateways
    # are stateless wrappers over settings, but a single registry instance
    # avoids constructing two separate NowPaymentsGateway/StripeGateway sets.
    gateway_registry = build_gateway_registry(settings)
    _services.payment_provider_state_service = PaymentProviderStateService(settings)
    _services.payment_service = PaymentService(
        billing_service=_services.billing_service,
        settings=settings,
        registry=gateway_registry,
        provider_state_service=_services.payment_provider_state_service,
    )
    logger.info("payment_service.initialized")

    # Initialize rate limiter
    init_rate_limiter(settings)

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
        logger.info(
            "r2.initialized",
            bucket=settings.r2_bucket_name,
            retention_days=settings.retention_days,
        )

        # Logo caching requires a dedicated *public* assets bucket — the
        # user-content bucket (r2_storage above) is private and served only
        # through the cookie-authenticated content proxy, so it can never
        # back publicly-linked logo_url values. When the assets bucket isn't
        # configured, logo caching is disabled but availability/metadata
        # sync still runs (logo_cache=None).
        logo_cache_service: LogoCacheService | None = None
        if settings.r2_public_assets_configured:
            assets_r2_settings = R2StorageSettings(
                account_id=settings.r2_account_id,
                access_key_id=settings.r2_access_key_id,
                secret_access_key=settings.r2_secret_access_key,
                bucket_name=settings.r2_public_assets_bucket or "",
            )
            _services.r2_public_assets_storage = R2StorageService(assets_r2_settings)
            logo_cache_service = LogoCacheService(r2_client=_services.r2_public_assets_storage)

            # Best-effort probe (P2-1): a misconfigured/scope-limited token
            # otherwise produces zero signal until the first sync tries (and
            # fails) to cache a logo. Fail-open — a broken assets bucket must
            # not block API startup; logo caching self-degrades per D10/P1-2.
            try:
                await _services.r2_public_assets_storage.exists(f"{LOGO_KEY_PREFIX}/.probe")
            except StorageError as exc:
                logger.exception(
                    "payment_currency.logo_storage_probe_failed",
                    bucket=settings.r2_public_assets_bucket,
                    error=str(exc),
                )
            else:
                logger.info(
                    "payment_currency.logo_cache_enabled",
                    bucket=settings.r2_public_assets_bucket,
                )
        else:
            logger.warning("payment_currency.logo_cache_disabled")

        _services.payment_currency_sync_service = PaymentCurrencySyncService(
            registry=gateway_registry,
            logo_cache=logo_cache_service,
        )
        logger.info("payment_currency_sync_service.initialized")

        if workers_enabled:
            _services.payment_currency_sync_worker = PaymentCurrencySyncWorker(
                db_manager=_services.db_manager,
                sync_service=_services.payment_currency_sync_service,
                interval=settings.payment_currency_sync_interval_seconds,
                redis_enabled=redis_enabled,
                redis_client_factory=get_operational_redis_client,
            )
            await _services.payment_currency_sync_worker.start()
            logger.info("payment_currency_sync_worker.started")
    else:
        logger.warning("r2.not_configured")

    # Initialize and start the content retention sweeper (requires the DB +
    # R2 initialized just above).
    if workers_enabled and settings.content_cleanup_enabled and _services.r2_storage is not None:
        retention_service = ContentRetentionService(
            session_factory=_services.db_manager.session_factory,
            storage=_services.r2_storage,
            batch_size=settings.content_cleanup_batch_size,
            max_batches_per_run=settings.content_cleanup_max_batches_per_run,
        )
        _services.content_retention_worker = ContentRetentionWorker(
            service=retention_service,
            interval=settings.content_cleanup_interval_seconds,
            redis_client_factory=get_operational_redis_client,
        )
        await _services.content_retention_worker.start()
        logger.info(
            "content_retention_worker.started",
            retention_days=settings.retention_days,
            interval=settings.content_cleanup_interval_seconds,
            batch_size=settings.content_cleanup_batch_size,
        )

    # Initialize and start the frame extraction worker (requires R2; not
    # gated behind any provider config flag — core capability).
    if workers_enabled and _services.r2_storage is not None:
        _services.frame_extraction_worker = FrameExtractionWorker(
            db_manager=_services.db_manager,
            r2_storage=_services.r2_storage,
            settings=settings,
            redis_enabled=redis_enabled,
            redis_client_factory=get_operational_redis_client,
        )
        await _services.frame_extraction_worker.start()
        logger.info("frame_extraction_worker.started")

    # Initialize Grok provider (if configured)
    if settings.grok_configured and _services.r2_storage is not None:
        grok_client = GrokClient(settings)
        await grok_client.connect()

        _services.grok_job_service = GrokJobService(
            grok_client=grok_client,
            storage=_services.r2_storage,
            retention_days=settings.retention_days,
            billing_service=get_billing_service(),
            event_bus=_services.event_bus,
            ops_event_bus=_services.ops_event_bus,
            max_poll_time=settings.grok_video_max_poll_time,
            finalization_lease_seconds=settings.grok_video_finalization_lease_seconds,
            billing_policy=ProviderBillingPolicyRegistry.with_grok_moderation_policy(
                settings.grok_moderation_billing_policy,
                settings.grok_undelivered_output_billing_policy,
            ),
        )
        await _services.grok_job_service.connect()
        logger.info("grok.initialized")
    else:
        logger.warning("grok.not_configured")

    # Start Grok video polling worker
    if workers_enabled and settings.grok_configured and _services.grok_job_service is not None:
        if settings.grok_video_worker_in_process:
            _services.grok_video_worker = GrokVideoWorker(
                db_manager=_services.db_manager,
                job_service=_services.grok_job_service,
                billing_service=get_billing_service(),
                settings=settings,
                event_bus=_services.event_bus,
                ops_event_bus=_services.ops_event_bus,
                redis_enabled=redis_enabled,
                redis_client_factory=get_operational_redis_client,
            )
            await _services.grok_video_worker.start()
            logger.info("grok.video_worker_in_process_started")
        else:
            logger.info(
                "grok.video_worker_external_recommended",
                hint=(
                    "Poll-on-read settles videos safely without a worker; run "
                    "'python -m src.workers.grok_video' for proactive completion"
                ),
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

    # Initialize email verification service. Depends on token_revocation_service
    # (issue #142 R1/A4) — must be constructed after it, which it already is
    # (see the token_revocation_service.initialized log above); the assertion
    # below guards against a future reordering silently reintroducing that bug.
    if _services.token_revocation_service is None:
        raise RuntimeError(
            "token_revocation_service must be initialized before email_verification_service"
        )
    _services.email_verification_service = EmailVerificationService(
        email_service=_services.email_service,
        app_url=settings.app_url,
        app_name=settings.app_name,
        token_revocation_service=_services.token_revocation_service,
        ops_event_bus=_services.ops_event_bus,
    )

    # Initialize and start Aisha job poller
    if workers_enabled and settings.aisha_poller_enabled:
        poller_config = AishaPollerConfig(
            enabled=True,
            tick_interval_seconds=settings.aisha_poller_tick_interval_seconds,
            max_concurrent_polls=settings.aisha_poller_max_concurrent_polls,
            job_age_warning_seconds=settings.aisha_poller_job_age_warning_seconds,
            job_age_timeout_seconds=settings.aisha_poller_job_age_timeout_seconds,
            comfyui_request_timeout_seconds=settings.aisha_poller_comfyui_request_timeout_seconds,
            tunnel_allowed_suffix=settings.aisha_cf_tunnel_domain or "",
            tunnel_allowed_prefix=settings.aisha_cf_tunnel_prefix or None,
            retention_days=settings.retention_days,
        )
        _services.aisha_job_poller = AishaJobPoller(
            session_factory=_services.db_manager.session_factory,
            event_bus=_services.event_bus,
            billing_service=get_billing_service(),
            r2_storage=_services.r2_storage,
            config=poller_config,
            ops_event_bus=_services.ops_event_bus,
            redis_enabled=redis_enabled,
            redis_client_factory=get_operational_redis_client,
        )
        await _services.aisha_job_poller.start()
        logger.info("aisha_job_poller.started")
    else:
        logger.info("aisha_job_poller.disabled")

    # Initialize GPU session stack (requires both Vast.ai + CF to be configured)
    if settings.vastai_configured and settings.aisha_cf_configured:
        from src.api.services.cloudflare.client import CloudflareTunnelClient
        from src.api.services.vastai.client import VastAIClient

        _services.gpu_session_http_client = httpx.AsyncClient(timeout=30.0)

        vastai_client = VastAIClient(
            http_client=_services.gpu_session_http_client,
            api_key=settings.vastai_api_key,
            settings=settings,
        )
        cf_client = CloudflareTunnelClient(
            http_client=_services.gpu_session_http_client,
            api_token=settings.aisha_cf_api_token,
            account_id=settings.aisha_cf_account_id,
            zone_id=settings.aisha_cf_zone_id,
            tunnel_domain=settings.aisha_cf_tunnel_domain,
        )
        _services.bundle_index = BundleIndexService(
            repo_url=settings.ai_bundles_repo_url,
            github_token=settings.ai_bundles_github_token,
            branch=settings.ai_bundles_branch,
            sync_interval_minutes=settings.ai_bundles_sync_interval_minutes,
            default_comfyui_port=settings.comfyui_port,
            max_download_bytes=settings.ai_bundles_max_download_bytes,
            max_member_count=settings.ai_bundles_max_member_count,
            max_member_size_bytes=settings.ai_bundles_max_member_size_bytes,
            max_uncompressed_bytes=settings.ai_bundles_max_uncompressed_bytes,
            settings=settings,
        )
        _services.workflow_service = WorkflowService(bundle_index=_services.bundle_index)
        _services.bundle_index.register_on_resync(_services.workflow_service.invalidate_cache)
        await _services.bundle_index.start()

        billing_service_for_worker = get_billing_service()

        job_sweep_service = JobSweepService(
            session_factory=_services.db_manager.session_factory,
            event_bus=_services.event_bus,
            billing_service=get_billing_service(),
            ops_event_bus=_services.ops_event_bus,
        )

        if settings.redis_url:
            from src.core.redis import get_redis_client

            cooldown_store: NodeCooldownStore = RedisNodeCooldownStore(get_redis_client(), settings)
        else:
            cooldown_store = NullNodeCooldownStore()

        _services.gpu_session_service = GpuSessionService(
            vastai_client=vastai_client,
            cf_client=cf_client,
            bundle_index=_services.bundle_index,
            session_factory=_services.db_manager.session_factory,
            settings=settings,
            billing_service=billing_service_for_worker,
            cooldown_store=cooldown_store,
            event_bus=_services.event_bus,
            job_sweep_service=job_sweep_service,
            ops_event_bus=_services.ops_event_bus,
        )

        if workers_enabled:
            _services.gpu_provisioning_worker = GpuProvisioningWorker(
                session_factory=_services.db_manager.session_factory,
                vastai_client=vastai_client,
                cf_client=cf_client,
                bundle_index=_services.bundle_index,
                http_client=_services.gpu_session_http_client,
                settings=settings,
                cooldown_store=cooldown_store,
                billing_service=billing_service_for_worker,
                event_bus=_services.event_bus,
                job_sweep_service=job_sweep_service,
                ops_event_bus=_services.ops_event_bus,
                redis_enabled=redis_enabled,
                redis_client_factory=get_operational_redis_client,
            )
            await _services.gpu_provisioning_worker.start()

            _services.orphaned_tunnel_cleanup_worker = OrphanedTunnelCleanupWorker(
                session_factory=_services.db_manager.session_factory,
                cf_client=cf_client,
                vastai_client=vastai_client,
                settings=settings,
                redis_enabled=redis_enabled,
                redis_client_factory=get_operational_redis_client,
            )
            await _services.orphaned_tunnel_cleanup_worker.start()

            if _services.gpu_session_service is None:
                raise RuntimeError(
                    "gpu_session_service must be initialized before billing_reconciler_worker"
                )
            _services.billing_reconciler_worker = BillingReconcilerWorker(
                session_factory=_services.db_manager.session_factory,
                gpu_session_service=_services.gpu_session_service,
                settings=settings,
                redis_enabled=redis_enabled,
                redis_client_factory=get_operational_redis_client,
            )
            await _services.billing_reconciler_worker.start()

        logger.info("gpu_session_stack.initialized")
    else:
        logger.warning(
            "gpu_session_stack.not_configured",
            hint="Set VASTAI_API_KEY and AISHA_CF_API_TOKEN/AISHA_CF_ACCOUNT_ID/AISHA_CF_ZONE_ID to enable GPU sessions",
        )

    # Initialize unified generation service
    from src.api.services.generation.aisha_provider import AishaGenerationProvider
    from src.api.services.generation.grok_provider import GrokGenerationProvider
    from src.core.enums import Provider

    generation_providers: dict[Provider, GenerationProvider] = {}
    if _services.bundle_index is None:
        logger.warning(
            "aisha_provider.disabled",
            reason="bundle index is not configured",
            hint="Ensure bundle indexing is set up if AISHA should be available",
        )
    elif _services.workflow_service is None:
        raise RuntimeError("workflow_service must be initialized before the Aisha provider")
    else:
        generation_providers[Provider.AISHA] = AishaGenerationProvider(
            workflow_service=_services.workflow_service,
            gpu_session_service=_services.gpu_session_service,
            bundle_index=_services.bundle_index,
            r2_storage=_services.r2_storage,
            tunnel_domain=settings.aisha_cf_tunnel_domain,
            max_input_megapixels=settings.image_max_input_megapixels,
        )

    if _services.grok_job_service is not None and _services.r2_storage is not None:
        generation_providers[Provider.GROK] = GrokGenerationProvider(
            grok_job_service=_services.grok_job_service,
            r2_storage=_services.r2_storage,
        )

    # Unified job service shares the same provider registry as GenerationService
    # so poll-on-read (refresh_job) dispatches through the same per-provider hooks.
    _services.unified_job_service = UnifiedJobService(providers=generation_providers)
    logger.info("unified_job_service.initialized")

    from src.api.services.generation.rate_limiter import ModelRateLimiter

    model_rate_limiter = ModelRateLimiter()

    _services.generation_service = GenerationService(
        providers=generation_providers,
        billing_service=get_billing_service(),
        pricing_service=get_pricing_service(),
        rate_limiter=model_rate_limiter,
        event_bus=_services.event_bus,
        ops_event_bus=_services.ops_event_bus,
        retention_days=settings.retention_days,
        bundle_index=_services.bundle_index,
    )
    logger.info(
        "generation_service.initialized",
        providers=[p.value for p in generation_providers],
    )

    # Initialize and start token cleanup worker
    if workers_enabled:
        _services.token_cleanup_worker = TokenCleanupWorker(
            db_manager=_services.db_manager,
            interval=settings.token_cleanup_interval_seconds,
            redis_enabled=redis_enabled,
            redis_client_factory=get_operational_redis_client,
        )
        await _services.token_cleanup_worker.start()
        logger.info("token_cleanup_worker.started")

    # Initialize Web Push (requires VAPID keys + Redis — see Settings.push_enabled)
    if settings.push_enabled:
        if settings.vapid_private_key is None or settings.vapid_subject is None:
            raise RuntimeError(
                "settings.push_enabled is True but vapid_private_key/vapid_subject is None"
            )
        _services.push_service = PushService(
            sender=PywebpushSender(
                private_key=settings.vapid_private_key,
                subject=settings.vapid_subject,
            ),
            broadcast_concurrency=settings.push_broadcast_concurrency,
        )
        logger.info("push_service.initialized")

        if workers_enabled:
            _services.push_dispatcher = PushDispatcher(
                push_service=_services.push_service,
                session_factory=_services.db_manager.session_factory,
                redis_enabled=redis_enabled,
            )
            await _services.push_dispatcher.start()
            logger.info("push_dispatcher.started")
    else:
        logger.warning(
            "push.not_configured",
            hint="Set VAPID_PUBLIC_KEY/VAPID_PRIVATE_KEY/VAPID_SUBJECT and REDIS_URL to enable Web Push",
        )

    # Initialize Telegram ops notifications (requires bot token + Redis — see
    # Settings.telegram_enabled)
    if settings.telegram_enabled:
        if settings.telegram_bot_token is None:
            raise RuntimeError("settings.telegram_enabled is True but telegram_bot_token is None")
        _services.telegram_sender = HttpxTelegramSender(
            bot_token=settings.telegram_bot_token.get_secret_value(),
            send_timeout_seconds=settings.telegram_send_timeout_seconds,
        )
        logger.info("telegram_sender.initialized")

        if workers_enabled:
            _services.telegram_dispatcher = TelegramDispatcher(
                sender=_services.telegram_sender,
                session_factory=_services.db_manager.session_factory,
                redis_enabled=redis_enabled,
            )
            await _services.telegram_dispatcher.start()
            logger.info("telegram.dispatcher.started")

            _services.telegram_link_poller = TelegramLinkPoller(
                sender=_services.telegram_sender,
                session_factory=_services.db_manager.session_factory,
                poll_timeout_seconds=settings.telegram_poll_timeout_seconds,
                redis_enabled=redis_enabled,
                redis_client_factory=get_operational_redis_client,
            )
            await _services.telegram_link_poller.start()
            logger.info("telegram.link_poller.started")
    else:
        logger.warning(
            "telegram.not_configured",
            hint="Set TELEGRAM_BOT_TOKEN and REDIS_URL to enable Telegram ops notifications",
        )

    _services.admin_notification_service = AdminNotificationService(
        sender=_services.telegram_sender,
        link_token_ttl_seconds=settings.telegram_link_token_ttl_seconds,
    )
    logger.info("admin_notification_service.initialized")

    # Initialize health check registry and service
    from src.api.services.health.checkers.cloud_provider import GrokChecker
    from src.api.services.health.checkers.gpu_session import GpuSessionReconciler
    from src.api.services.health.checkers.infrastructure import (
        PostgresChecker,
        R2Checker,
        RedisChecker,
    )
    from src.api.services.health.checkers.platform_api import VastAIChecker
    from src.api.services.health.checkers.token_revocation import TokenRevocationChecker
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
        health_registry.register(RedisChecker(redis=get_operational_redis_client()))
    if _services.r2_storage is not None:
        health_registry.register(R2Checker(r2_storage=_services.r2_storage))
    if _services.token_revocation_service is None:
        raise RuntimeError("token_revocation_service must be initialized before health checks")
    # Registered unconditionally (not gated on settings.redis_url) — the
    # checker itself reports `inactive` with an explanatory message when
    # Redis isn't configured (A2), so that state is visible on the health
    # endpoint rather than simply absent from it.
    health_registry.register(TokenRevocationChecker(service=_services.token_revocation_service))

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
            tunnel_domain=settings.aisha_cf_tunnel_domain,
            tunnel_prefix=settings.aisha_cf_tunnel_prefix,
            grace_seconds=settings.gpu_session_stale_grace_seconds,
        )
    )

    _services.health_service = HealthService(
        registry=health_registry,
        timeout_seconds=settings.health_check_timeout_seconds,
    )
    logger.info("health_service.initialized")

    # Build credit guard only when the GPU session stack is available.
    session_credit_guard: SessionCreditGuard | None = None
    if _services.gpu_session_service is not None:
        session_credit_guard = SessionCreditGuard(
            session_factory=_services.db_manager.session_factory,
            billing_service=get_billing_service(),
            gpu_session_service=_services.gpu_session_service,
            settings=settings,
            event_bus=_services.event_bus,
        )
        logger.info("session_credit_guard.initialized")

    # Initialize and start health snapshot + snapshot-cleanup workers
    if workers_enabled:
        _services.health_snapshot_worker = HealthSnapshotWorker(
            health_service=_services.health_service,
            db_manager=_services.db_manager,
            interval_seconds=settings.health_snapshot_interval_seconds,
            redis_url=settings.redis_url,
            session_credit_guard=session_credit_guard,
            ops_event_bus=_services.ops_event_bus,
            redis_client_factory=get_operational_redis_client,
        )
        await _services.health_snapshot_worker.start()

        _services.health_snapshot_cleanup_worker = HealthSnapshotCleanupWorker(
            db_manager=_services.db_manager,
            retention_days=settings.health_snapshot_retention_days,
            redis_enabled=redis_enabled,
            redis_client_factory=get_operational_redis_client,
        )
        await _services.health_snapshot_cleanup_worker.start()

    # Register the worker heartbeat checker last, now that every worker that
    # will ever exist this process has been constructed. health_registry is
    # held by reference inside health_service, so registering after the
    # fact still takes effect for future checks.
    all_workers: list[PeriodicWorker] = [
        w
        for w in (
            _services.token_cleanup_worker,
            _services.content_retention_worker,
            _services.frame_extraction_worker,
            _services.aisha_job_poller,
            _services.gpu_provisioning_worker,
            _services.orphaned_tunnel_cleanup_worker,
            _services.billing_reconciler_worker,
            _services.health_snapshot_worker,
            _services.health_snapshot_cleanup_worker,
            _services.grok_video_worker,
            _services.payment_currency_sync_worker,
            _services.telegram_link_poller,
        )
        if w is not None
    ]
    health_registry.register(WorkerHeartbeatChecker(workers=all_workers))

    return _services.jwt_service  # Return for storing in app.state


async def shutdown_services() -> None:
    """Cleanup service resources.

    Called during application shutdown.
    """
    global _services

    # Stop GPU session workers before closing their dependencies
    if _services.gpu_provisioning_worker is not None:
        await _services.gpu_provisioning_worker.stop()

    if _services.billing_reconciler_worker is not None:
        await _services.billing_reconciler_worker.stop()

    if _services.orphaned_tunnel_cleanup_worker is not None:
        await _services.orphaned_tunnel_cleanup_worker.stop()

    # Stop the bundle-index sync loop. Owned by the GPU session stack — its
    # background asyncio task must be cancelled before the event loop closes
    # to avoid pending-task warnings on shutdown.
    if _services.bundle_index is not None:
        await _services.bundle_index.stop()
        logger.info("bundle_index.stopped")

    if _services.gpu_session_http_client is not None:
        await _services.gpu_session_http_client.aclose()
        logger.info("gpu_session_http_client.closed")

    if _services.aisha_job_poller is not None:
        await _services.aisha_job_poller.stop()
        logger.info("aisha_job_poller.stopped")

    # Stop DB-dependent workers before closing the pool so their cancellation
    # cleanup (rollback + session.close) can still return connections cleanly.
    if _services.token_cleanup_worker is not None:
        await _services.token_cleanup_worker.stop()

    if _services.content_retention_worker is not None:
        await _services.content_retention_worker.stop()

    if _services.payment_currency_sync_worker is not None:
        await _services.payment_currency_sync_worker.stop()

    if _services.frame_extraction_worker is not None:
        await _services.frame_extraction_worker.stop()

    if _services.push_dispatcher is not None:
        await _services.push_dispatcher.stop()

    if _services.telegram_dispatcher is not None:
        await _services.telegram_dispatcher.stop()

    if _services.telegram_link_poller is not None:
        await _services.telegram_link_poller.stop()

    if _services.telegram_sender is not None:
        await _services.telegram_sender.aclose()

    if _services.health_snapshot_worker is not None:
        await _services.health_snapshot_worker.stop()

    if _services.health_snapshot_cleanup_worker is not None:
        await _services.health_snapshot_cleanup_worker.stop()

    if _services.grok_video_worker is not None:
        await _services.grok_video_worker.stop()

    if _services.r2_storage is not None:
        await _services.r2_storage.close()
        logger.info("r2.closed")

    if _services.r2_public_assets_storage is not None:
        await _services.r2_public_assets_storage.close()
        logger.info("r2_public_assets.closed")

    if _services.db_manager is not None:
        await _services.db_manager.close()
        logger.info("db.closed")

    if _services.grok_job_service is not None:
        await _services.grok_job_service.close()
        logger.info("grok.closed")

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
    "auth_service": Provide(get_auth_service, sync_to_thread=False),
    "user_service": Provide(get_user_service, sync_to_thread=False),
    # Core services
    "workflow_service": Provide(get_workflow_service, sync_to_thread=False),
    "settings": Provide(provide_settings, sync_to_thread=False),
    # Storage services
    "r2_storage": Provide(get_r2_storage, sync_to_thread=False),
    "session": Provide(get_db_session),
    "user_content": Provide(get_user_content, sync_to_thread=False),
    # Video frame extraction
    "frame_extraction_service": Provide(get_frame_extraction_service, sync_to_thread=False),
    # Grok services
    "grok_job_service": Provide(get_grok_job_service, sync_to_thread=False),
    # Billing services
    "billing_service": Provide(get_billing_service, sync_to_thread=False),
    "pricing_service": Provide(get_pricing_service, sync_to_thread=False),
    "organization_service": Provide(get_organization_service, sync_to_thread=False),
    "payment_service": Provide(get_payment_service, sync_to_thread=False),
    "payment_provider_state_service": Provide(
        get_payment_provider_state_service, sync_to_thread=False
    ),
    "payment_currency_sync_service": Provide(
        get_payment_currency_sync_service, sync_to_thread=False
    ),
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
    # JWT service (needed by auth routes to mint content tokens)
    "jwt_service": Provide(get_jwt_service, sync_to_thread=False),
    # Token revocation (needed by the logout route to denylist its own jti)
    "token_revocation_service": Provide(get_token_revocation_service, sync_to_thread=False),
    # Content proxy
    "content_proxy": Provide(get_content_proxy, sync_to_thread=False),
    # Library
    "library_service": Provide(get_library_service, sync_to_thread=False),
    "library_project_service": Provide(get_library_project_service, sync_to_thread=False),
    "library_tag_service": Provide(get_library_tag_service, sync_to_thread=False),
    # Idempotency
    "idempotency_service": Provide(get_idempotency_service, sync_to_thread=False),
    # Health
    "health_service": Provide(get_health_service, sync_to_thread=False),
    # GPU sessions
    "gpu_session_service": Provide(get_gpu_session_service, sync_to_thread=False),
    # Internal operation events (node bearer auth validated in handler)
    "operation_event_service": Provide(get_operation_event_service, sync_to_thread=False),
    # Web Push
    "push_service": Provide(get_push_service, sync_to_thread=False),
    # Admin ops notifications (Telegram)
    "admin_notification_service": Provide(get_admin_notification_service, sync_to_thread=False),
}
