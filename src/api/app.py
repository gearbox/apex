"""Litestar application factory and configuration."""

import re
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import structlog
from litestar import Litestar, Request, Response
from litestar.config.cors import CORSConfig
from litestar.datastructures import UploadFile
from litestar.exceptions import HTTPException
from litestar.middleware import DefineMiddleware
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Contact, Server
from litestar.status_codes import (
    HTTP_400_BAD_REQUEST,
    HTTP_402_PAYMENT_REQUIRED,
    HTTP_403_FORBIDDEN,
    HTTP_404_NOT_FOUND,
    HTTP_409_CONFLICT,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_500_INTERNAL_SERVER_ERROR,
)

from src.api.dependencies.common import dependencies, init_services, shutdown_services
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.product import ProductMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware, build_rate_limit_config
from src.api.responses import error_response as _error
from src.api.routes.admin import AdminController
from src.api.routes.admin_management import AdminManagementController
from src.api.routes.auth import AuthController
from src.api.routes.billing import BillingController, BillingWebhookController
from src.api.routes.billing_public import BillingPublicController
from src.api.routes.content import ContentProxyController
from src.api.routes.gallery import GalleryController
from src.api.routes.gpu_session import GpuSessionController
from src.api.routes.health import AdminHealthController, HealthController
from src.api.routes.internal_gpu_session import InternalGpuSessionController
from src.api.routes.jobs import UnifiedJobController
from src.api.routes.organization import OrganizationController
from src.api.routes.payment_provider_admin import PaymentProviderAdminController
from src.api.routes.providers import ProvidersController
from src.api.routes.push import PushController
from src.api.routes.sse import SSEController
from src.api.routes.storage import StorageController
from src.api.routes.unified_generation import UnifiedGenerationController
from src.api.routes.user import UserController
from src.api.schemas.errors import ErrorEnvelope
from src.api.services.billing_errors import (
    AccountInactiveError,
    AccountNotFoundError,
    InsufficientBalanceError,
    ModerationError,
    OrganizationBalanceError,
    OrganizationPermissionError,
    PaymentProviderDisabledError,
    PaymentVerificationError,
    PriceNotFoundError,
    RefundNotEligibleError,
)
from src.api.services.idempotency import IdempotencyConflictError
from src.core.config import Settings, get_settings
from src.core.logging import configure_logging
from src.core.product_registry import PRODUCT_REGISTRY

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Exception handlers — all produce a unified ErrorEnvelope
# ---------------------------------------------------------------------------

_STATUS_TO_ERROR_CODE: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    409: "conflict",
    422: "validation_error",
    429: "too_many_requests",
    500: "internal_error",
    503: "service_unavailable",
}


def _log_handler_event(
    event: str,
    request: Request[Any, Any, Any],
    status_code: int,
    level: str = "warning",
    exc_info: BaseException | None = None,
    **extra: object,  # These are structlog kwargs
) -> None:
    """Emit a structured log event with the consistent field set for all exception handlers.

    Every handler call includes path, method, and status_code. Extra domain-specific
    fields (e.g. balance, provider) are forwarded via **extra.
    """
    getattr(logger, level)(
        event,
        path=request.url.path,
        method=request.method,
        status_code=status_code,
        exc_info=exc_info,
        **extra,
    )


def http_exception_handler(request: Request[Any, Any, Any], exc: HTTPException) -> Response[Any]:
    error_code = _STATUS_TO_ERROR_CODE.get(exc.status_code, "error")
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    if exc.status_code >= 500:
        _log_handler_event(
            "http.error", request, exc.status_code, level="error", exc_info=exc, error=error_code
        )
    else:
        _log_handler_event("http.error", request, exc.status_code, error=error_code)
    return _error(error_code, message, exc.status_code)


def insufficient_balance_handler(
    request: Request[Any, Any, Any],
    exc: InsufficientBalanceError,
) -> Response[Any]:
    _log_handler_event(
        "billing.insufficient_balance",
        request,
        HTTP_402_PAYMENT_REQUIRED,
        balance=exc.balance,
        required=exc.required,
    )
    return _error(
        "insufficient_balance",
        str(exc),
        HTTP_402_PAYMENT_REQUIRED,
        {"balance": exc.balance, "required": exc.required},
    )


def account_not_found_handler(
    request: Request[Any, Any, Any],
    exc: AccountNotFoundError,
) -> Response[Any]:
    _log_handler_event("billing.account_not_found", request, HTTP_404_NOT_FOUND)
    return _error("account_not_found", str(exc), HTTP_404_NOT_FOUND)


def account_inactive_handler(
    request: Request[Any, Any, Any],
    exc: AccountInactiveError,
) -> Response[Any]:
    _log_handler_event("billing.account_inactive", request, HTTP_403_FORBIDDEN)
    return _error("account_inactive", str(exc), HTTP_403_FORBIDDEN)


def refund_not_eligible_handler(
    request: Request[Any, Any, Any],
    exc: RefundNotEligibleError,
) -> Response[Any]:
    _log_handler_event("billing.refund_not_eligible", request, HTTP_409_CONFLICT)
    return _error("refund_not_eligible", str(exc), HTTP_409_CONFLICT)


def price_not_found_handler(
    request: Request[Any, Any, Any],
    exc: PriceNotFoundError,
) -> Response[Any]:
    _log_handler_event("billing.price_not_found", request, HTTP_404_NOT_FOUND)
    return _error("price_not_found", str(exc), HTTP_404_NOT_FOUND)


def moderation_error_handler(
    request: Request[Any, Any, Any],
    exc: ModerationError,
) -> Response[Any]:
    _log_handler_event(
        "moderation.rejected",
        request,
        HTTP_422_UNPROCESSABLE_ENTITY,
        provider=exc.provider,
        policy=exc.policy,
    )
    return _error(
        "moderation",
        str(exc),
        HTTP_422_UNPROCESSABLE_ENTITY,
        {"provider": exc.provider, "policy": exc.policy},
    )


def payment_verification_handler(
    request: Request[Any, Any, Any],
    exc: PaymentVerificationError,
) -> Response[Any]:
    _log_handler_event("payment.verification_failed", request, HTTP_400_BAD_REQUEST)
    return _error("payment_verification_failed", str(exc), HTTP_400_BAD_REQUEST)


def payment_provider_disabled_handler(
    request: Request[Any, Any, Any],
    exc: PaymentProviderDisabledError,
) -> Response[Any]:
    _log_handler_event(
        "payment.provider_disabled",
        request,
        HTTP_409_CONFLICT,
        provider=exc.provider.value,
    )
    return Response(
        content={"code": "payment_provider_disabled", "provider": exc.provider.value},
        status_code=HTTP_409_CONFLICT,
    )


def organization_permission_handler(
    request: Request[Any, Any, Any],
    exc: OrganizationPermissionError,
) -> Response[Any]:
    _log_handler_event("organization.permission_denied", request, HTTP_403_FORBIDDEN)
    return _error("permission_denied", str(exc), HTTP_403_FORBIDDEN)


def organization_balance_handler(
    request: Request[Any, Any, Any],
    exc: OrganizationBalanceError,
) -> Response[Any]:
    _log_handler_event(
        "organization.balance_nonzero",
        request,
        HTTP_409_CONFLICT,
        balance=exc.balance,
    )
    return _error(
        "organization_balance_nonzero",
        str(exc),
        HTTP_409_CONFLICT,
        {"balance": exc.balance},
    )


def idempotency_conflict_handler(
    request: Request[Any, Any, Any],
    exc: IdempotencyConflictError,
) -> Response[Any]:
    _log_handler_event("idempotency.conflict", request, HTTP_409_CONFLICT, level="info")
    return Response(
        content=ErrorEnvelope(
            error="idempotency_conflict",
            message=str(exc),
            status_code=HTTP_409_CONFLICT,
        ),
        status_code=HTTP_409_CONFLICT,
        headers={"Retry-After": "1"},
    )


def global_exception_handler(request: Request[Any, Any, Any], exc: Exception) -> Response[Any]:
    """Catch-all for any unhandled exception.

    Returns a generic ErrorEnvelope and logs the full traceback at error level.
    Never leaks internal details (exception class name, message, stack) to the client.

    Note: asyncio.CancelledError, KeyboardInterrupt, and SystemExit inherit from
    BaseException (not Exception) since Python 3.9 and cannot reach this handler.
    """
    _log_handler_event(
        "unhandled_exception",
        request,
        HTTP_500_INTERNAL_SERVER_ERROR,
        level="error",
        exc_info=exc,
        exc_type=type(exc).__qualname__,
    )
    return _error(
        "internal_error",
        "An unexpected error occurred.",
        HTTP_500_INTERNAL_SERVER_ERROR,
    )


def _build_openapi_config(settings: Settings) -> OpenAPIConfig:
    """Build the OpenAPI schema/docs configuration.

    Only called when ``settings.enable_docs`` is true — see ``create_app()``.
    """
    return OpenAPIConfig(
        title="Apex Generation API",
        version="0.3.1",
        description=(
            "Apex REST API for AI content generation.\n\n"
            "## Providers\n\n"
            "- **Aisha**: Custom workflow-based generation\n"
            "- **Grok**: xAI's image and video generation\n\n"
            "## Authentication\n\n"
            "Authentication is handled via OAuth2/JWT (implementation pending).\n"
        ),
        contact=Contact(name="API Support"),
        servers=[
            Server(
                url=f"http://{settings.api_host}:{settings.api_port}",
                description="Local development server",
            ),
        ],
        path="/docs",
    )


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncGenerator[None]:
    """Application lifespan manager.

    Initializes services on startup and cleans up on shutdown.
    """
    settings = get_settings()
    configure_logging(settings)

    logger.info("app.startup")

    jwt_service = await init_services(settings)

    # Store JWT service in app state for auth guards
    app.state["jwt_service"] = jwt_service

    try:
        yield
    finally:
        logger.info("app.shutdown")
        await shutdown_services()


def create_app() -> Litestar:
    # sourcery skip: inline-immediately-returned-variable
    """Create and configure Litestar application.

    Returns:
        Configured Litestar application instance.
    """
    settings = get_settings()
    configure_logging(settings)

    # CORS — derive allowed origins from the product registry so we never
    # hard-code brand strings here. Wildcard is invalid with credentials.
    _domains = sorted({d for pc in PRODUCT_REGISTRY.values() for d in pc.domains})
    _escaped = "|".join(re.escape(d) for d in _domains)
    if settings.debug:
        _origin_regex = rf"^(https://([a-z0-9-]+\.)?({_escaped})|http://localhost(:\d+)?)$"
    else:
        _origin_regex = rf"^https://([a-z0-9-]+\.)?({_escaped})$"

    cors_config = CORSConfig(
        allow_origins=[],  # must be empty — origin matching is done by allow_origin_regex
        allow_origin_regex=_origin_regex,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=[
            "Authorization",
            "Content-Type",
            "X-Product-Id",
            "Idempotency-Key",
            "X-Request-Id",
        ],
    )

    # OpenAPI documentation — off by default; see Settings.enable_docs.
    openapi_config = _build_openapi_config(settings) if settings.enable_docs else None

    return Litestar(
        route_handlers=[
            HealthController,
            AdminHealthController,
            # Authentication (public)
            AuthController,
            # User management (authenticated)
            UserController,
            # Billing & payments
            BillingController,
            BillingPublicController,
            BillingWebhookController,
            # Organizations
            OrganizationController,
            # Admin
            AdminController,
            AdminManagementController,
            PaymentProviderAdminController,
            # Generation (unified)
            UnifiedGenerationController,  # POST /v1/generate
            ProvidersController,  # GET /v1/providers
            # GPU Sessions
            GpuSessionController,  # /v1/sessions/*
            InternalGpuSessionController,  # /v1/internal/gpu-sessions/*
            # Storage
            StorageController,
            # Jobs (unified, cross-provider)
            UnifiedJobController,
            # Real-time events (SSE)
            SSEController,
            # Gallery
            GalleryController,
            # Content proxy
            ContentProxyController,
            # Web Push
            PushController,
        ],
        exception_handlers={
            HTTPException: http_exception_handler,
            InsufficientBalanceError: insufficient_balance_handler,
            AccountNotFoundError: account_not_found_handler,
            AccountInactiveError: account_inactive_handler,
            RefundNotEligibleError: refund_not_eligible_handler,
            PriceNotFoundError: price_not_found_handler,
            ModerationError: moderation_error_handler,
            PaymentVerificationError: payment_verification_handler,
            PaymentProviderDisabledError: payment_provider_disabled_handler,
            OrganizationPermissionError: organization_permission_handler,
            OrganizationBalanceError: organization_balance_handler,
            IdempotencyConflictError: idempotency_conflict_handler,
            Exception: global_exception_handler,
        },
        dependencies=dependencies,
        lifespan=[lifespan],
        middleware=[
            ProductMiddleware,
            RequestLoggingMiddleware,
            DefineMiddleware(
                RateLimitMiddleware,
                config=build_rate_limit_config(settings),
            ),
        ],
        cors_config=cors_config,
        openapi_config=openapi_config,
        debug=settings.debug,
        signature_types=[UploadFile],
        request_max_body_size=settings.max_upload_size_bytes,
    )


# Application instance
app = create_app()
