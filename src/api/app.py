"""Litestar application factory and configuration."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from litestar import Litestar, Request, Response
from litestar.config.cors import CORSConfig
from litestar.datastructures import UploadFile
from litestar.logging import LoggingConfig
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
from src.api.routes.admin import AdminController
from src.api.routes.auth import AuthController
from src.api.routes.billing import BillingController, BillingWebhookController
from src.api.routes.generation import (
    GenerationController,
    HealthController,
    ImageController,
    JobController,
)
from src.api.routes.grok import (
    GrokImageController,
    GrokJobController,
    GrokProviderController,
    GrokVideoController,
)
from src.api.routes.organization import OrganizationController
from src.api.routes.storage import StorageController
from src.api.routes.user import UserController
from src.api.services.billing_errors import (
    AccountInactiveError,
    AccountNotFoundError,
    InsufficientBalanceError,
    ModerationError,
    OrganizationPermissionError,
    PaymentVerificationError,
    PriceNotFoundError,
    RefundNotEligibleError,
)
from src.core.config import get_settings

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exception handlers for billing errors
# ---------------------------------------------------------------------------


def _billing_error_response(
    request: Request,  # noqa: ARG001
    exc: Exception,
    status_code: int,
) -> Response:
    return Response(content={"detail": str(exc)}, status_code=status_code)


def insufficient_balance_handler(request: Request, exc: InsufficientBalanceError) -> Response:  # noqa: ARG001
    return Response(
        content={"detail": str(exc), "balance": exc.balance, "required": exc.required},
        status_code=HTTP_402_PAYMENT_REQUIRED,
    )


def account_not_found_handler(request: Request, exc: AccountNotFoundError) -> Response:
    return _billing_error_response(request, exc, HTTP_404_NOT_FOUND)


def account_inactive_handler(request: Request, exc: AccountInactiveError) -> Response:
    return _billing_error_response(request, exc, HTTP_403_FORBIDDEN)


def refund_not_eligible_handler(request: Request, exc: RefundNotEligibleError) -> Response:
    return _billing_error_response(request, exc, HTTP_409_CONFLICT)


def price_not_found_handler(request: Request, exc: PriceNotFoundError) -> Response:
    return _billing_error_response(request, exc, HTTP_404_NOT_FOUND)


def moderation_error_handler(request: Request, exc: ModerationError) -> Response:  # noqa: ARG001
    return Response(
        content={"detail": str(exc), "provider": exc.provider, "policy": exc.policy},
        status_code=HTTP_422_UNPROCESSABLE_ENTITY,
    )


def payment_verification_handler(request: Request, exc: PaymentVerificationError) -> Response:
    return _billing_error_response(request, exc, HTTP_400_BAD_REQUEST)


def organization_permission_handler(request: Request, exc: OrganizationPermissionError) -> Response:
    return _billing_error_response(request, exc, HTTP_403_FORBIDDEN)


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncGenerator[None]:  # noqa: ARG001
    """Application lifespan manager.

    Initializes services on startup and cleans up on shutdown.
    """
    settings = get_settings()

    # Determine base path for workflows
    # Check common locations
    possible_paths = [
        Path.cwd(),  # Current directory
        Path(__file__).parent.parent.parent.parent,  # Project root
        Path("/app"),  # Container path
    ]

    base_path = None
    for path in possible_paths:
        workflow_check = path / "config" / "bundles"
        if workflow_check.exists():
            base_path = path
            logger.info(f"Found workflow bundles at: {base_path}")
            break

    if base_path is None:
        logger.warning("Workflow bundles directory not found, using current directory")
        base_path = Path.cwd()

    logger.info(f"Starting Apex API service, connecting to {settings.comfyui_base_url}")

    jwt_service = await init_services(settings, base_path=base_path)

    # Store JWT service in app state for auth guards
    app.state["jwt_service"] = jwt_service

    try:
        yield
    finally:
        logger.info("Shutting down Apex API service")
        await shutdown_services()


def create_app() -> Litestar:
    # sourcery skip: inline-immediately-returned-variable
    """Create and configure Litestar application.

    Returns:
        Configured Litestar application instance.
    """
    settings = get_settings()

    # CORS configuration for development
    cors_config = CORSConfig(
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Logging configuration
    logging_config = LoggingConfig(
        root={
            "level": "DEBUG" if settings.debug else "INFO",
            "handlers": ["console"],
        },
        formatters={
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            },
        },
        handlers={
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
            },
        },
        loggers={
            "src": {
                "level": "DEBUG" if settings.debug else "INFO",
                "propagate": True,
            },
            "httpx": {
                "level": "WARNING",
                "propagate": False,
            },
            "xai_sdk": {
                "level": "INFO",
                "propagate": False,
            },
        },
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
            # Generation (TODO: add auth_guard later)
            GenerationController,
            JobController,
            ImageController,
            # Storage (TODO: add auth_guard later)
            StorageController,
            # Grok generation
            GrokProviderController,
            GrokImageController,
            GrokVideoController,
            GrokJobController,
        ],
        exception_handlers={
            InsufficientBalanceError: insufficient_balance_handler,
            AccountNotFoundError: account_not_found_handler,
            AccountInactiveError: account_inactive_handler,
            RefundNotEligibleError: refund_not_eligible_handler,
            PriceNotFoundError: price_not_found_handler,
            ModerationError: moderation_error_handler,
            PaymentVerificationError: payment_verification_handler,
            OrganizationPermissionError: organization_permission_handler,
        },
        dependencies=dependencies,
        lifespan=[lifespan],
        cors_config=cors_config,
        logging_config=logging_config,
        openapi_config=openapi_config,
        debug=settings.debug,
        signature_types=[UploadFile],
    )

    return app


# Application instance for uvicorn
app = create_app()
