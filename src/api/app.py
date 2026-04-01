"""Litestar application factory and configuration."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
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
)

from src.api.dependencies.common import dependencies, init_services, shutdown_services
from src.api.middleware.logging import RequestLoggingMiddleware
from src.api.middleware.product import ProductMiddleware
from src.api.middleware.rate_limit import RateLimitMiddleware, build_rate_limit_config
from src.api.routes.admin import AdminController
from src.api.routes.auth import AuthController
from src.api.routes.billing import BillingController, BillingWebhookController
from src.api.routes.content import ContentProxyController
from src.api.routes.gallery import GalleryController
from src.api.routes.generation import ImageController
from src.api.routes.health import AdminHealthController, HealthController
from src.api.routes.jobs import UnifiedJobController
from src.api.routes.organization import OrganizationController
from src.api.routes.providers import ProvidersController
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
    PaymentVerificationError,
    PriceNotFoundError,
    RefundNotEligibleError,
)
from src.api.services.idempotency import IdempotencyConflictError
from src.core.config import get_settings
from src.core.logging import configure_logging

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


def _error(
    error: str, message: str, status_code: int, detail: dict[str, Any] | None = None
) -> Response[Any]:
    return Response(
        content=ErrorEnvelope(error=error, message=message, status_code=status_code, detail=detail),
        status_code=status_code,
    )


def http_exception_handler(request: Request[Any, Any, Any], exc: HTTPException) -> Response[Any]:  # noqa: ARG001
    error_code = _STATUS_TO_ERROR_CODE.get(exc.status_code, "error")
    message = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return _error(error_code, message, exc.status_code)


def insufficient_balance_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: InsufficientBalanceError,
) -> Response[Any]:
    return _error(
        "insufficient_balance",
        str(exc),
        HTTP_402_PAYMENT_REQUIRED,
        {"balance": exc.balance, "required": exc.required},
    )


def account_not_found_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: AccountNotFoundError,
) -> Response[Any]:
    return _error("account_not_found", str(exc), HTTP_404_NOT_FOUND)


def account_inactive_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: AccountInactiveError,
) -> Response[Any]:
    return _error("account_inactive", str(exc), HTTP_403_FORBIDDEN)


def refund_not_eligible_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: RefundNotEligibleError,
) -> Response[Any]:
    return _error("refund_not_eligible", str(exc), HTTP_409_CONFLICT)


def price_not_found_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: PriceNotFoundError,
) -> Response[Any]:
    return _error("price_not_found", str(exc), HTTP_404_NOT_FOUND)


def moderation_error_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: ModerationError,
) -> Response[Any]:
    return _error(
        "moderation",
        str(exc),
        HTTP_422_UNPROCESSABLE_ENTITY,
        {"provider": exc.provider, "policy": exc.policy},
    )


def payment_verification_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: PaymentVerificationError,
) -> Response[Any]:
    return _error("payment_verification_failed", str(exc), HTTP_400_BAD_REQUEST)


def organization_permission_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: OrganizationPermissionError,
) -> Response[Any]:
    return _error("permission_denied", str(exc), HTTP_403_FORBIDDEN)


def organization_balance_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: OrganizationBalanceError,
) -> Response[Any]:
    return _error(
        "organization_balance_nonzero",
        str(exc),
        HTTP_409_CONFLICT,
        {"balance": exc.balance},
    )


def idempotency_conflict_handler(
    request: Request[Any, Any, Any],  # noqa: ARG001
    exc: IdempotencyConflictError,
) -> Response[Any]:
    return Response(
        content=ErrorEnvelope(
            error="idempotency_conflict",
            message=str(exc),
            status_code=HTTP_409_CONFLICT,
        ),
        status_code=HTTP_409_CONFLICT,
        headers={"Retry-After": "1"},
    )


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncGenerator[None]:  # noqa: ARG001
    """Application lifespan manager.

    Initializes services on startup and cleans up on shutdown.
    """
    settings = get_settings()
    configure_logging(settings)

    # Determine base path for workflows
    # Check common locations
    possible_paths = [
        Path.cwd(),  # Current directory
        Path(__file__).parent.parent.parent.parent,  # Project root
        Path("/app"),  # Container path
    ]

    base_path = None
    for base_path in possible_paths:
        workflow_check = base_path / "config" / "bundles"
        if workflow_check.exists():
            logger.info("app.bundles_found", path=str(base_path))
            break
    else:
        # Loop completed without break: use first path as fallback
        logger.warning("app.bundles_not_found")
        base_path = possible_paths[0]  # Use current directory as fallback

    logger.info("app.startup", comfyui_url=settings.comfyui_base_url)

    jwt_service = await init_services(settings, base_path=base_path)

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

    # CORS configuration for development
    cors_config = CORSConfig(
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # OpenAPI documentation configuration
    openapi_config = OpenAPIConfig(
        title="Apex Generation API",
        version="0.2.0",
        description=(
            "Apex REST API for AI content generation.\n\n"
            "## Providers\n\n"
            "- **ComfyUI**: Custom workflow-based generation (T2I, I2I)\n"
            "- **Grok**: xAI's image and video generation (T2I, I2I, T2V, I2V)\n\n"
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

    app = Litestar(
        route_handlers=[
            HealthController,
            AdminHealthController,
            # Authentication (public)
            AuthController,
            # User management (authenticated)
            UserController,
            # Billing & payments
            BillingController,
            BillingWebhookController,
            # Organizations
            OrganizationController,
            # Admin
            AdminController,
            # Generation (unified)
            UnifiedGenerationController,  # POST /v1/generate
            ProvidersController,  # GET /v1/providers
            ImageController,
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
            OrganizationPermissionError: organization_permission_handler,
            OrganizationBalanceError: organization_balance_handler,
            IdempotencyConflictError: idempotency_conflict_handler,
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
    )

    return app


# Application instance
app = create_app()
