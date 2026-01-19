"""Litestar application factory and configuration."""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path

from litestar import Litestar
from litestar.config.cors import CORSConfig
from litestar.datastructures import UploadFile
from litestar.logging import LoggingConfig
from litestar.openapi import OpenAPIConfig
from litestar.openapi.spec import Contact, Server

from src.api.dependencies import dependencies, init_services, shutdown_services
from src.api.routes import (
    GenerationController,
    HealthController,
    ImageController,
    JobController,
    StorageController,
)
from src.core.config import get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: Litestar) -> AsyncGenerator[None, None]:  # noqa: ARG001
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

    await init_services(settings, base_path=base_path)

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
        },
    )

    # OpenAPI documentation configuration
    openapi_config = OpenAPIConfig(
        title="Apex Generation API",
        version="0.1.0",
        description="Apex REST API for ComfyUI generation workflows",
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
            GenerationController,
            JobController,
            ImageController,
            StorageController,
        ],
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
