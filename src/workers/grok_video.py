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
from typing import TYPE_CHECKING, NoReturn

import structlog

from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.grok import GrokClient
from src.api.services.grok.factory import build_grok_job_service
from src.api.services.video_jobs.cli import BuiltVideoService, VideoWorkerDeps, run_video_worker
from src.core.config import Settings, get_settings
from src.core.logging import configure_logging

if TYPE_CHECKING:
    from src.api.services.grok.job_service import GrokJobService

logger = structlog.get_logger(__name__)

WORKER_NAME = "grok_video_poller"
"""Feeds the leader-lease key — must stay byte-for-byte stable across releases."""


class GrokVideoWorkerCLI:
    """CLI runner for the Grok video polling worker."""

    def __init__(self, settings: Settings) -> None:
        """Initialize the CLI runner.

        Args:
            settings: Application settings.
        """
        self._settings = settings
        self._shutdown_event = asyncio.Event()

    async def run(self) -> None:
        """Run the video worker until shutdown signal."""
        await run_video_worker(
            self._settings,
            shutdown_event=self._shutdown_event,
            build_job_service=self._build_job_service,
            worker_name=WORKER_NAME,
            interval_seconds=self._settings.grok_video_poll_interval,
            max_concurrent_polls=self._settings.grok_video_max_concurrent_polls,
            required=lambda settings: settings.grok_configured,
        )

    @staticmethod
    async def _build_job_service(deps: VideoWorkerDeps) -> BuiltVideoService:
        """Connect the Grok client and job service this process polls with."""
        settings = deps.settings
        grok_client = GrokClient(settings)
        await grok_client.connect()
        logger.info("grok_worker.client_initialized")

        job_service: GrokJobService = build_grok_job_service(
            grok_client=grok_client,
            storage=deps.storage,
            media_ingestor=deps.media_ingestor,
            billing_service=deps.billing_service,
            event_bus=deps.event_bus,
            ops_event_bus=deps.ops_event_bus,
            billing_policy=ProviderBillingPolicyRegistry.from_settings(settings),
            retention_days=settings.retention_days,
            max_poll_time=settings.grok_video_max_poll_time,
            finalization_lease_seconds=settings.grok_video_finalization_lease_seconds,
        )
        await job_service.connect()

        async def close() -> None:
            await job_service.close()
            await grok_client.close()

        return BuiltVideoService(service=job_service, close=close)

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
