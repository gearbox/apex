"""CLI commands for Apex workers.

Run as:
    python -m src.workers.grok_video

Or with uvx/uv:
    uv run python -m src.workers.grok_video
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import NoReturn

from src.api.services.grok import GrokClient
from src.api.services.grok.job_service import GrokJobService
from src.api.services.grok.video_worker import GrokVideoWorker
from src.api.services.storage import R2StorageService, R2StorageSettings
from src.core.config import Settings, get_settings
from src.db import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GrokVideoWorkerCLI:
    """CLI runner for the Grok video polling worker."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the CLI runner.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._worker: GrokVideoWorker | None = None
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Run the video worker until shutdown signal."""
        logger.info("Starting Grok video worker...")

        # Validate configuration
        if not self._settings.grok_configured:
            logger.error("Grok API key not configured (XAI_API_KEY)")
            sys.exit(1)

        if not self._settings.r2_configured:
            logger.error("R2 storage not configured")
            sys.exit(1)

        # Initialize database
        db_manager = init_db(
            self._settings.database_url,
            pool_size=self._settings.db_pool_size,
            max_overflow=self._settings.db_max_overflow,
            echo=self._settings.db_echo,
        )
        logger.info("Database connection pool initialized")

        # Initialize R2 storage
        r2_settings = R2StorageSettings(
            account_id=self._settings.r2_account_id,
            access_key_id=self._settings.r2_access_key_id,
            secret_access_key=self._settings.r2_secret_access_key,
            bucket_name=self._settings.r2_bucket_name,
            public_url_base=self._settings.r2_public_url_base,
            retention_days=self._settings.retention_days,
        )
        r2_storage = R2StorageService(r2_settings)
        logger.info(f"R2 storage initialized for bucket: {self._settings.r2_bucket_name}")

        # Initialize Grok client
        grok_client = GrokClient(self._settings)
        await grok_client.connect()
        logger.info("Grok client initialized")

        # Initialize job service
        job_service = GrokJobService(
            grok_client=grok_client,
            storage=r2_storage,
            retention_days=self._settings.retention_days,
        )
        await job_service.connect()
        logger.info("Grok job service initialized")

        # Create and start worker
        self._worker = GrokVideoWorker(
            db_manager=db_manager,
            job_service=job_service,
            settings=self._settings,
        )
        await self._worker.start()

        logger.info(
            f"Grok video worker running "
            f"(poll interval: {self._settings.grok_video_poll_interval}s, "
            f"max poll time: {self._settings.grok_video_max_poll_time}s)"
        )
        logger.info("Press Ctrl+C to stop")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Cleanup
        logger.info("Shutting down...")
        await self._worker.stop()
        await job_service.close()
        await grok_client.close()
        await r2_storage.close()
        await db_manager.close()
        logger.info("Grok video worker stopped")

    def handle_signal(self) -> None:
        """Handle shutdown signal."""
        logger.info("Received shutdown signal")
        self._shutdown_event.set()


def main() -> NoReturn:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Grok video polling worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  XAI_API_KEY              xAI API key (required)
  DATABASE_URL             PostgreSQL connection URL
  R2_ACCOUNT_ID            Cloudflare R2 account ID
  R2_ACCESS_KEY_ID         R2 access key
  R2_SECRET_ACCESS_KEY     R2 secret key
  R2_BUCKET_NAME           R2 bucket name
  GROK_VIDEO_POLL_INTERVAL Seconds between polls (default: 5)
  GROK_VIDEO_MAX_POLL_TIME Maximum poll time in seconds (default: 600)
        """,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    # Configure logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("src").setLevel(logging.DEBUG)

    # Load settings
    settings = get_settings()

    # Create runner
    runner = GrokVideoWorkerCLI(settings)

    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, runner.handle_signal)

    try:
        loop.run_until_complete(runner.run())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()

    sys.exit(0)


if __name__ == "__main__":
    main()
