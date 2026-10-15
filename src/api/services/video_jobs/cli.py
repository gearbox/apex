"""Shared body of the standalone async-video worker processes.

Each provider's entry point (``python -m src.workers.<provider>_video``)
supplies only what differs: how to build
its job service, its worker name/interval/concurrency and its config check.
"""

from __future__ import annotations

import asyncio
import dataclasses
import sys
from typing import TYPE_CHECKING

import structlog

from src.api.services.billing import BillingService
from src.api.services.event_bus import EventBus
from src.api.services.media_ingest.factory import (
    build_media_ingest_service,
    build_provider_video_ingest_service,
)
from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.storage import R2StorageService, R2StorageSettings
from src.api.services.video_jobs.receipts import build_accepted_task_receipts
from src.api.services.video_jobs.reconciler import (
    AcceptedTaskReconciler,
    AcceptedTaskReconcilerWorker,
)
from src.api.services.video_jobs.worker import AsyncVideoJobWorker
from src.core.redis import get_operational_redis_client
from src.db import init_db

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from src.api.services.media_ingest import MediaIngestor
    from src.api.services.video_jobs.protocol import AsyncVideoJobService, OrphanTaskCompensator
    from src.api.services.video_jobs.receipts import AcceptedTaskReceipts
    from src.core.config import Settings

logger = structlog.get_logger(__name__)


@dataclasses.dataclass(frozen=True, slots=True)
class VideoWorkerDeps:
    """Process-level collaborators handed to a provider's job-service builder."""

    settings: Settings
    storage: R2StorageService
    billing_service: BillingService
    event_bus: EventBus
    ops_event_bus: OpsEventBus
    media_ingestor: MediaIngestor
    provider_video_ingestor: MediaIngestor
    """Prepares provider-generated videos under ``MEDIA_INGEST_PROVIDER_VIDEO_MAX_MB``."""
    receipts: AcceptedTaskReceipts
    """Accepted-task receipt store on the dedicated receipt pool."""


@dataclasses.dataclass(frozen=True, slots=True)
class BuiltVideoService:
    """A connected job service plus the teardown for everything it opened."""

    service: AsyncVideoJobService
    close: Callable[[], Awaitable[None]]
    compensator: OrphanTaskCompensator
    """The provider's cancel/observe adapter, driven by the receipt reconciler."""
    receipt_reconciler_name: str
    """Receipt reconciler worker (and leader-lease) name."""


async def run_video_worker(
    settings: Settings,
    *,
    shutdown_event: asyncio.Event,
    build_job_service: Callable[[VideoWorkerDeps], Awaitable[BuiltVideoService]],
    worker_name: str,
    interval_seconds: int,
    max_concurrent_polls: int,
    required: Callable[[Settings], bool],
) -> None:
    """Run one provider's video worker until ``shutdown_event`` is set.

    Args:
        settings: Application settings.
        shutdown_event: Set by the caller's signal handler.
        build_job_service: Builds and connects the provider job service.
        worker_name: Worker (and leader-lease) name.
        interval_seconds: Poll interval.
        max_concurrent_polls: Semaphore bound.
        required: True when the provider is configured; otherwise the process exits 1.
    """
    logger.info("video_worker.starting", worker=worker_name)

    # Validate configuration
    if not required(settings):
        logger.error("video_worker.missing_api_key", worker=worker_name)
        sys.exit(1)

    if not settings.r2_configured:
        logger.error("video_worker.r2_not_configured", worker=worker_name)
        sys.exit(1)

    # Initialize database
    db_manager = init_db(
        settings.database_url,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        echo=settings.db_echo,
    )
    logger.info("video_worker.db_initialized", worker=worker_name)

    # The shared pool remains necessary for EventBus/OpsEventBus publish;
    # the operational pool is dedicated to this worker's leader lease.
    if settings.redis_url:
        from src.core.redis import init_operational_redis_pool, init_redis_pool

        init_redis_pool(
            settings.redis_url,
            socket_connect_timeout=settings.redis_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_socket_timeout_seconds,
            health_check_interval=settings.redis_health_check_interval_seconds,
            max_connections=settings.redis_max_connections,
        )
        init_operational_redis_pool(
            settings.redis_url,
            socket_connect_timeout=settings.redis_operational_socket_connect_timeout_seconds,
            socket_timeout=settings.redis_operational_socket_timeout_seconds,
            health_check_interval=settings.redis_health_check_interval_seconds,
            max_connections=settings.redis_operational_max_connections,
        )
        logger.info("video_worker.redis_initialized", worker=worker_name)

    # Initialize R2 storage
    r2_settings = R2StorageSettings(
        account_id=settings.r2_account_id,
        access_key_id=settings.r2_access_key_id,
        secret_access_key=settings.r2_secret_access_key,
        bucket_name=settings.r2_bucket_name,
        retention_days=settings.retention_days,
    )
    r2_storage = R2StorageService(r2_settings)
    logger.info("video_worker.r2_initialized", worker=worker_name, bucket=settings.r2_bucket_name)

    # Billing service is stateless — see src.api.services.billing docstring.
    billing_service = BillingService()
    # Match the in-process topology.  Disabled instances are intentional
    # no-ops, so standalone polling remains safe without Redis.
    event_bus = EventBus(enabled=settings.redis_url is not None)
    ops_event_bus = OpsEventBus(enabled=settings.redis_url is not None)
    receipt_db, receipts = build_accepted_task_receipts(settings, ops_event_bus)

    # One admission budget for the process: both ingest services share the semaphores.
    video_slots = asyncio.Semaphore(settings.media_ingest_video_concurrency)
    image_slots = asyncio.Semaphore(settings.media_ingest_image_concurrency)

    built = await build_job_service(
        VideoWorkerDeps(
            settings=settings,
            storage=r2_storage,
            billing_service=billing_service,
            event_bus=event_bus,
            ops_event_bus=ops_event_bus,
            media_ingestor=build_media_ingest_service(
                settings, video_slots=video_slots, image_slots=image_slots
            ),
            provider_video_ingestor=build_provider_video_ingest_service(
                settings, video_slots=video_slots, image_slots=image_slots
            ),
            receipts=receipts,
        )
    )
    logger.info("video_worker.job_service_initialized", worker=worker_name)

    # redis_enabled lets this standalone process share the same leader lease
    # key as any in-process worker, so running both simultaneously is safe
    # (only one actually ticks).
    worker = AsyncVideoJobWorker(
        name=worker_name,
        job_service=built.service,
        interval_seconds=interval_seconds,
        max_concurrent_polls=max_concurrent_polls,
        db_manager=db_manager,
        event_bus=event_bus,
        redis_enabled=settings.redis_url is not None,
        redis_client_factory=get_operational_redis_client,
    )
    await worker.start()

    # Orphan reconciliation shares the lease with an in-process twin, like the poller.
    reconciler_worker = AcceptedTaskReconcilerWorker(
        name=built.receipt_reconciler_name,
        reconciler=AcceptedTaskReconciler(
            receipt_db,
            built.compensator,
            ops_event_bus,
            grace_seconds=settings.video_task_receipt_grace_seconds,
        ),
        interval_seconds=settings.video_task_receipt_reconcile_interval_seconds,
        redis_enabled=settings.redis_url is not None,
        redis_client_factory=get_operational_redis_client,
    )
    await reconciler_worker.start()

    logger.info(
        "video_worker.running",
        worker=worker_name,
        poll_interval=interval_seconds,
        max_concurrent_polls=max_concurrent_polls,
    )
    logger.info("video_worker.ready", worker=worker_name, hint="Press Ctrl+C to stop")

    # Wait for shutdown signal
    await shutdown_event.wait()

    # Cleanup
    logger.info("video_worker.shutting_down", worker=worker_name)
    await reconciler_worker.stop()
    await worker.stop()
    await built.close()
    await r2_storage.close()
    await receipt_db.close()
    await db_manager.close()
    if settings.redis_url:
        from src.core.redis import close_redis_pool

        await close_redis_pool()
    logger.info("video_worker.stopped", worker=worker_name)
