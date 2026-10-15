"""Standalone Wan (Alibaba Model Studio) video polling worker.

Run as:
    python -m src.workers.wan_video

Or with uvx/uv:
    uv run python -m src.workers.wan_video

Independent of ``python -m src.workers.grok_video``: its own leader lease
(``alibaba_video_poller``), interval and concurrency, so the two providers
poll and finalize at the same time without affecting each other.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from typing import NoReturn

import structlog

from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
from src.api.services.media_ingest.profile_log import VideoProfileOrigin
from src.api.services.model_studio.client import ModelStudioClient
from src.api.services.model_studio.orphan_compensator import ModelStudioOrphanCompensator
from src.api.services.model_studio.wan_job_service import WanJobService
from src.api.services.video_jobs.cli import BuiltVideoService, VideoWorkerDeps, run_video_worker
from src.api.services.video_jobs.download import VideoDownloadClient
from src.api.services.video_jobs.materializer import RemoteVideoMaterializer
from src.api.services.video_jobs.transition import make_transition_service_factory
from src.core.config import Settings, get_settings
from src.core.enums import Provider
from src.core.logging import configure_logging

logger = structlog.get_logger(__name__)

WORKER_NAME = WanJobService.VIDEO_POLLER_NAME
"""Feeds the leader-lease key; defined once on the job service (shared with the in-process worker)."""


class WanVideoPollerCLI:
    """CLI runner for the Wan video polling worker."""

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
            interval_seconds=self._settings.wan_video_poll_interval,
            max_concurrent_polls=self._settings.wan_video_max_concurrent_polls,
            required=lambda settings: settings.model_studio_configured,
        )

    @staticmethod
    async def _build_job_service(deps: VideoWorkerDeps) -> BuiltVideoService:
        """Connect the Model Studio client and Wan job service this process polls with."""
        settings = deps.settings
        if settings.model_studio_api_key is None:  # guarded by ``required``; narrows the type
            raise RuntimeError("MODEL_STUDIO_API_KEY is not set")
        client = ModelStudioClient(
            base_url=settings.model_studio_base_url,
            api_key=settings.model_studio_api_key,
            timeout_seconds=settings.model_studio_timeout_seconds,
        )
        await client.connect()
        logger.info("wan_worker.client_initialized")

        policy = ProviderBillingPolicyRegistry.from_settings(settings)
        download = VideoDownloadClient()
        materializer = RemoteVideoMaterializer(
            provider=Provider.ALIBABA,
            profile_origin=VideoProfileOrigin.ALIBABA_OUTPUT,
            storage=deps.storage,
            media_ingestor=deps.provider_video_ingestor,
            http_client_provider=lambda: download.client,
            retention_days=settings.retention_days,
            finalization_lease_seconds=settings.wan_video_finalization_lease_seconds,
            max_output_bytes=settings.media_ingest_provider_video_max_bytes,
            billing_policy=policy,
            transition_service_factory=make_transition_service_factory(
                billing_service=deps.billing_service,
                event_bus=deps.event_bus,
                ops_event_bus=deps.ops_event_bus,
            ),
        )
        compensator = ModelStudioOrphanCompensator(client)
        job_service = WanJobService(
            client,
            materializer,
            download,
            receipts=deps.receipts,
            compensator=compensator,
            billing_service=deps.billing_service,
            event_bus=deps.event_bus,
            ops_event_bus=deps.ops_event_bus,
            max_poll_time=settings.wan_video_max_poll_time,
            billing_policy=policy,
        )
        await job_service.connect()

        async def close() -> None:
            await job_service.close()
            await client.close()

        return BuiltVideoService(
            service=job_service,
            close=close,
            compensator=compensator,
            receipt_reconciler_name=WanJobService.RECEIPT_RECONCILER_NAME,
        )

    def handle_signal(self) -> None:
        """Handle shutdown signal."""
        logger.info("wan_worker.signal_received")
        self._shutdown_event.set()


def main() -> NoReturn:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Wan (Alibaba Model Studio) video polling worker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment variables:
  MODEL_STUDIO_BASE_URL    Model Studio workspace host (required)
  MODEL_STUDIO_API_KEY     Model Studio API key (required)
  DATABASE_URL             PostgreSQL connection URL
  R2_ACCOUNT_ID            Cloudflare R2 account ID
  R2_ACCESS_KEY_ID         R2 access key
  R2_SECRET_ACCESS_KEY     R2 secret key
  R2_BUCKET_NAME           R2 bucket name
  WAN_VIDEO_POLL_INTERVAL  Seconds between polls (default: 15)
  WAN_VIDEO_MAX_POLL_TIME  Maximum poll time in seconds (default: 1800)
        """,
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings)
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    runner = WanVideoPollerCLI(settings)

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
