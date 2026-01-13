"""Main entry point for running the API server."""

import uvicorn

from src.core.config import get_settings


def main() -> None:
    """Run the API server with uvicorn."""
    settings = get_settings()

    uvicorn.run(
        "src.api.app:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
