"""Main entry point for running the API server."""

import granian
from granian.constants import Interfaces

from src.core.config import get_settings
from src.core.logging import configure_logging


def main() -> None:
    """Run the API server"""
    settings = get_settings()
    configure_logging(settings)

    granian.Granian(
        target="src.api.app:app",
        address=settings.api_host,
        port=settings.api_port,
        interface=Interfaces.ASGI,
        workers=settings.asgi_workers,  # default 1
        reload=settings.debug,
        log_enabled=False,  # structlog handles everything
    ).serve()


if __name__ == "__main__":
    main()
