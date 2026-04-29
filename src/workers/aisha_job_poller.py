"""Background worker: polls ComfyUI for in-progress Aisha generation jobs.

Started in app lifespan. One tick per ``config.tick_interval_seconds`` (default 1s).

Per tick:
  1. Query all QUEUED/RUNNING Aisha jobs whose GPU session is ACTIVE.
  2. Fan out concurrently (bounded by semaphore) — one ComfyUI HTTP round-trip
     per job.
  3. Drive state transitions through JobStateTransitionService — DB writes and
     event publishing in one atomic operation per job.

The poller does NOT write to the DB directly and does NOT handle billing.
All writes go through JobStateTransitionService.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.generation.tunnel_validation import (
    InvalidTunnelHostnameError,
    validate_tunnel_hostname,
)
from src.api.services.job_state_transition import (
    GenerationOutputData,
    JobStateTransitionService,
)
from src.api.services.storage import R2StorageService, StorageType
from src.core.enums import GpuSessionStatus, JobStatus
from src.db.models.storage import GenerationJob as GenerationJobModel
from src.db.repositories.job import JobRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.billing import BillingService
    from src.api.services.event_bus import EventBus
    from src.db.models.gpu_session import GpuSession
    from src.db.models.storage import GenerationJob

logger = structlog.get_logger(__name__)

_CONTENT_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


@dataclasses.dataclass
class AishaPollerConfig:
    """Runtime configuration for AishaJobPoller."""

    enabled: bool = True
    tick_interval_seconds: float = 1.0
    max_concurrent_polls: int = 16
    job_age_warning_seconds: int = 300
    job_age_timeout_seconds: int = 1800
    comfyui_request_timeout_seconds: float = 5.0
    tunnel_allowed_suffix: str = ""
    retention_days: int = 7


class AishaJobPoller:
    """Polls ComfyUI for in-progress Aisha generation jobs and drives transitions.

    Mirrors GpuProvisioningWorker: started from on_startup, owns a tick loop,
    acquires its own DB session per tick and per job poll.
    """

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        event_bus: EventBus | None,
        billing_service: BillingService,
        r2_storage: R2StorageService | None,
        config: AishaPollerConfig,
    ) -> None:
        self._session_factory = session_factory
        self._event_bus = event_bus
        self._billing = billing_service
        self._r2 = r2_storage
        self._config = config
        self._running = False
        self._task: asyncio.Task[None] | None = None

    def _make_transition_service(self, session: AsyncSession) -> JobStateTransitionService:
        return JobStateTransitionService(
            session=session,
            event_bus=self._event_bus,
            billing_service=self._billing,
        )

    async def start(self) -> None:
        """Spawn the polling loop. Idempotent."""
        if not self._config.enabled:
            logger.info("aisha_job_poller.disabled")
            return
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info(
            "aisha_job_poller.started",
            tick_interval_s=self._config.tick_interval_seconds,
            max_concurrent=self._config.max_concurrent_polls,
        )

    async def stop(self) -> None:
        """Cancel the loop and await graceful drain. Idempotent."""
        if not self._running:
            return
        self._running = False
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
        logger.info("aisha_job_poller.stopped")

    # -------------------------------------------------------------------------
    # Internal loop
    # -------------------------------------------------------------------------

    async def _loop(self) -> None:
        while self._running:
            try:
                await self._tick()
            except Exception:
                logger.exception("aisha_job_poller.tick_error")
            await asyncio.sleep(self._config.tick_interval_seconds)

    async def _tick(self) -> None:
        async with self._session_factory() as session:
            repo = JobRepository(session)
            jobs = list(await repo.list_aisha_jobs_for_polling(limit=200))

        if not jobs:
            return

        sem = asyncio.Semaphore(self._config.max_concurrent_polls)

        async def guarded(job_id: UUID, gpu_session_id: UUID) -> None:
            async with sem:
                await self._poll_one_with_own_session(job_id, gpu_session_id)

        await asyncio.gather(
            *(guarded(j.id, j.gpu_session_id) for j in jobs if j.gpu_session_id is not None),
            return_exceptions=True,
        )

    async def _poll_one_with_own_session(
        self,
        job_id: UUID,
        gpu_session_id: UUID,
    ) -> None:
        try:
            async with self._session_factory() as session:
                from sqlalchemy import select as sa_select
                from sqlalchemy.orm import selectinload

                result = await session.execute(
                    sa_select(GenerationJobModel)
                    .where(GenerationJobModel.id == job_id)
                    .options(selectinload(GenerationJobModel.gpu_session))
                )
                job = result.scalar_one_or_none()

                if job is None:
                    return

                if job.gpu_session is None:
                    logger.warning(
                        "aisha_job_poller.no_gpu_session",
                        job_id=str(job_id),
                    )
                    return

                await self._poll_one(job, job.gpu_session, session)
        except Exception:
            logger.exception(
                "aisha_job_poller.poll_error",
                job_id=str(job_id),
                gpu_session_id=str(gpu_session_id),
            )

    async def _poll_one(
        self,
        job: GenerationJob,
        gpu_session: GpuSession,
        session: AsyncSession,
    ) -> None:
        """Poll ComfyUI for one job and drive the appropriate state transition.

        Does NOT write to the DB directly — all writes go through
        JobStateTransitionService. Does NOT call BillingService — billing is
        settled inside the transition.
        """
        job_id = job.id
        product_id = str(job.product_id)

        # 1. Check session status — skip if not active
        if gpu_session.status != GpuSessionStatus.active.value:
            logger.debug(
                "aisha_job_poller.session_not_active",
                job_id=str(job_id),
                gpu_session_id=str(gpu_session.id),
                status=gpu_session.status,
            )
            return

        # 2. SSRF guard — validate tunnel hostname
        hostname = gpu_session.tunnel_hostname or ""
        allowed_suffix = (
            f".{self._config.tunnel_allowed_suffix}" if self._config.tunnel_allowed_suffix else ""
        )
        try:
            validate_tunnel_hostname(hostname, allowed_suffix=allowed_suffix)
        except InvalidTunnelHostnameError as exc:
            logger.error(
                "aisha_job_poller.invalid_tunnel_hostname",
                job_id=str(job_id),
                gpu_session_id=str(gpu_session.id),
                hostname=hostname,
                error=str(exc),
            )
            ts = JobStateTransitionService(
                session=session,
                event_bus=self._event_bus,
                billing_service=self._billing,
            )
            await ts.transition_to_failed(
                job_id,
                error_message="Generation session has an invalid tunnel hostname.",
                refund=True,
                product_id=product_id,
            )
            return

        # 3. Build per-request ComfyUI client (cheap, do not cache)
        prompt_id = job.external_request_id
        if not prompt_id:
            logger.warning(
                "aisha_job_poller.no_prompt_id",
                job_id=str(job_id),
            )
            return

        base_url = f"https://{hostname}"
        client = ComfyUIClient(
            base_url,
            request_timeout=self._config.comfyui_request_timeout_seconds,
        )

        # 4. Poll history + queue
        try:
            await client.connect()
            history = await client.get_history(prompt_id)

            if prompt_id in history:
                await self._handle_history_complete(
                    client=client,
                    job=job,
                    history_entry=history[prompt_id],
                    session=session,
                    product_id=product_id,
                )
            else:
                queue = await client.get_queue()
                await self._handle_queue_state(
                    job=job,
                    queue=queue,
                    prompt_id=prompt_id,
                    session=session,
                    product_id=product_id,
                )
        finally:
            await client.close()

    # -------------------------------------------------------------------------
    # History / queue handlers
    # -------------------------------------------------------------------------

    async def _handle_history_complete(
        self,
        *,
        client: ComfyUIClient,
        job: GenerationJob,
        history_entry: dict[str, Any],
        session: AsyncSession,
        product_id: str,
    ) -> None:
        if "outputs" not in history_entry:
            # History entry exists but no outputs — check for error flag
            status_info = history_entry.get("status", {})
            if isinstance(status_info, dict) and status_info.get("status_str") == "error":
                messages = status_info.get("messages", [["Error", "Unknown"]])
                ts = JobStateTransitionService(
                    session=session,
                    event_bus=self._event_bus,
                    billing_service=self._billing,
                )
                await ts.transition_to_failed(
                    job.id,
                    error_message=f"ComfyUI reported error: {messages!r}"[:500],
                    refund=True,
                    product_id=product_id,
                )
            return

        image_infos = self._collect_image_infos(history_entry)
        outputs: list[GenerationOutputData] = []
        expires_at = JobStateTransitionService.make_output_expires_at(self._config.retention_days)

        for idx, img_info in enumerate(image_infos):
            out_data = await self._download_and_upload(
                client=client,
                job=job,
                img_info=img_info,
                output_index=idx,
                expires_at=expires_at,
            )
            if out_data is not None:
                outputs.append(out_data)

        ts = JobStateTransitionService(
            session=session,
            event_bus=self._event_bus,
            billing_service=self._billing,
        )
        await ts.transition_to_completed(
            job.id,
            outputs=outputs,
            product_id=product_id,
        )

    async def _handle_queue_state(
        self,
        *,
        job: GenerationJob,
        queue: dict[str, Any],
        prompt_id: str,
        session: AsyncSession,
        product_id: str,
    ) -> None:
        running = queue.get("queue_running", [])
        is_running = any(item[1] == prompt_id for item in running if len(item) > 1)

        if is_running and str(job.status) == JobStatus.QUEUED.value:
            ts = JobStateTransitionService(
                session=session,
                event_bus=self._event_bus,
                billing_service=self._billing,
            )
            await ts.transition_to_running(job.id)
            return

        # Absent from both history and queue — check age-based timeout
        if job.started_at is None:
            return

        age = datetime.now(UTC) - job.started_at.replace(tzinfo=UTC)

        if age.total_seconds() >= self._config.job_age_warning_seconds:
            logger.warning(
                "aisha_job_poller.job_age_warning",
                job_id=str(job.id),
                age_seconds=int(age.total_seconds()),
            )

        if age.total_seconds() >= self._config.job_age_timeout_seconds:
            ts = JobStateTransitionService(
                session=session,
                event_bus=self._event_bus,
                billing_service=self._billing,
            )
            await ts.transition_to_failed(
                job.id,
                error_message="ComfyUI lost track of prompt — job timed out.",
                refund=True,
                product_id=product_id,
            )
        else:
            logger.debug(
                "aisha_job_poller.prompt_not_found_transient",
                job_id=str(job.id),
                age_seconds=int(age.total_seconds()),
            )

    # -------------------------------------------------------------------------
    # Image download + R2 upload
    # -------------------------------------------------------------------------

    async def _download_and_upload(
        self,
        *,
        client: ComfyUIClient,
        job: GenerationJob,
        img_info: dict[str, Any],
        output_index: int,
        expires_at: datetime,
    ) -> GenerationOutputData | None:
        if self._r2 is None:
            logger.warning(
                "aisha_job_poller.r2_not_configured",
                job_id=str(job.id),
            )
            return None

        filename: str = img_info.get("filename", "")
        subfolder: str = img_info.get("subfolder", "")
        img_type: str = img_info.get("type", "output")
        if not filename:
            return None

        try:
            data = await client.get_image(
                filename=filename,
                subfolder=subfolder,
                folder_type=img_type,
            )
        except Exception:
            logger.warning(
                "aisha_job_poller.get_image_failed",
                job_id=str(job.id),
                filename=filename,
            )
            return None

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        content_type = _CONTENT_TYPES.get(ext, "image/png")

        try:
            result = await self._r2.upload(
                user_id=job.user_id,
                data=data,
                filename=filename,
                content_type=content_type,
                storage_type=StorageType.OUTPUT,
                job_id=job.id,
            )
        except Exception:
            logger.warning(
                "aisha_job_poller.r2_upload_failed",
                job_id=str(job.id),
                filename=filename,
            )
            return None

        return GenerationOutputData(
            id=result.id,
            storage_key=result.storage_key,
            content_type=content_type,
            size_bytes=len(data),
            format=ext,
            output_index=output_index,
            expires_at=expires_at,
        )

    @staticmethod
    def _collect_image_infos(history_entry: dict[str, Any]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for _node_id, node_output in history_entry.get("outputs", {}).items():
            if "images" in node_output:
                images.extend(node_output["images"])
        return images
