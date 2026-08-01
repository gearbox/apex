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

import structlog

from src.api.services.billing import BillingService
from src.api.services.event_bus import EventBus
from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.grok import GrokClient
from src.api.services.grok.job_service import GrokJobService
from src.api.services.grok.video_worker import GrokVideoWorker
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.storage import R2StorageService, R2StorageSettings
from src.core.config import Settings, get_settings
from src.core.logging import configure_logging
from src.db import init_db

logger = structlog.get_logger(__name__)


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
        logger.info("grok_worker.starting")

        # Validate configuration
        if not self._settings.grok_configured:
            logger.error("grok_worker.missing_api_key")
            sys.exit(1)

        if not self._settings.r2_configured:
            logger.error("grok_worker.r2_not_configured")
            sys.exit(1)

        # Initialize database
        db_manager = init_db(
            self._settings.database_url,
            pool_size=self._settings.db_pool_size,
            max_overflow=self._settings.db_max_overflow,
            echo=self._settings.db_echo,
        )
        logger.info("grok_worker.db_initialized")

        # Redis for the leader lease — lets this standalone process and any
        # in-process worker share ownership safely (see GrokVideoWorker init).
        if self._settings.redis_url:
            from src.core.redis import init_redis_pool

            init_redis_pool(
                self._settings.redis_url,
                socket_connect_timeout=self._settings.redis_socket_connect_timeout_seconds,
                socket_timeout=self._settings.redis_socket_timeout_seconds,
                health_check_interval=self._settings.redis_health_check_interval_seconds,
                max_connections=self._settings.redis_max_connections,
            )
            logger.info("grok_worker.redis_initialized")

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
        logger.info("grok_worker.r2_initialized", bucket=self._settings.r2_bucket_name)

        # Initialize Grok client
        grok_client = GrokClient(self._settings)
        await grok_client.connect()
        logger.info("grok_worker.client_initialized")

        # Billing service is stateless — see src.api.services.billing docstring.
        billing_service = BillingService()
        # Match the in-process topology.  Disabled instances are intentional
        # no-ops, so standalone polling remains safe without Redis.
        event_bus = EventBus(enabled=self._settings.redis_url is not None)
        ops_event_bus = OpsEventBus(enabled=self._settings.redis_url is not None)

        # Initialize job service
        job_service = GrokJobService(
            grok_client=grok_client,
            storage=r2_storage,
            retention_days=self._settings.retention_days,
            billing_service=billing_service,
            event_bus=event_bus,
            ops_event_bus=ops_event_bus,
            max_poll_time=self._settings.grok_video_max_poll_time,
            finalization_lease_seconds=self._settings.grok_video_finalization_lease_seconds,
            billing_policy=ProviderBillingPolicyRegistry.with_grok_moderation_policy(
                self._settings.grok_moderation_billing_policy
            ),
        )
        await job_service.connect()
        logger.info("grok_worker.job_service_initialized")

        # Create and start worker. redis_enabled lets this standalone process
        # share the same leader lease key as any in-process worker, so
        # running both simultaneously is safe (only one actually ticks).
        self._worker = GrokVideoWorker(
            db_manager=db_manager,
            job_service=job_service,
            billing_service=billing_service,
            settings=self._settings,
            event_bus=event_bus,
            ops_event_bus=ops_event_bus,
            redis_enabled=self._settings.redis_url is not None,
        )
        await self._worker.start()

        logger.info(
            "grok_worker.running",
            poll_interval=self._settings.grok_video_poll_interval,
            max_poll_time=self._settings.grok_video_max_poll_time,
            finalization_lease_seconds=self._settings.grok_video_finalization_lease_seconds,
            moderation_billing_policy=self._settings.grok_moderation_billing_policy,
        )
        logger.info("grok_worker.ready", hint="Press Ctrl+C to stop")

        # Wait for shutdown signal
        await self._shutdown_event.wait()

        # Cleanup
        logger.info("grok_worker.shutting_down")
        await self._worker.stop()
        await job_service.close()
        await grok_client.close()
        await r2_storage.close()
        await db_manager.close()
        if self._settings.redis_url:
            from src.core.redis import close_redis_pool

            await close_redis_pool()
        logger.info("grok_worker.stopped")

    def handle_signal(self) -> None:
        """Handle shutdown signal."""
        logger.info("grok_worker.signal_received")
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

    # Load settings
    settings = get_settings()

    # Configure structlog (respects LOG_LEVEL / LOG_FORMAT env vars)
    configure_logging(settings)

    # Override log level when --debug flag is passed
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

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
