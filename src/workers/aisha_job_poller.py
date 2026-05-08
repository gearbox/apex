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

import httpx
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
from src.core.enums import JobStatus
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

        if suffix := (config.tunnel_allowed_suffix or "").strip():
            self._allowed_tunnel_suffix = suffix if suffix.startswith(".") else f".{suffix}"
        else:
            raise ValueError(
                "AishaPollerConfig.tunnel_allowed_suffix is empty. "
                "If Aisha is intentionally disabled in this environment, "
                "set AISHA_POLLER_ENABLED=false. Otherwise, set "
                "AISHA_CF_TUNNEL_DOMAIN to the GPU tunnel domain "
                "(e.g. 'gpu.cloudin.space')."
            )

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

        async def guarded(job: GenerationJob) -> None:
            if job.gpu_session is None:
                # Should not happen: the repo query joins gpu_session and the
                # CHECK constraint guarantees Aisha jobs have a session.
                logger.warning(
                    "aisha_job_poller.no_gpu_session_for_aisha_job",
                    job_id=str(job.id),
                )
                return
            async with sem:
                try:
                    async with self._session_factory() as job_session:
                        await self._poll_one(job, job.gpu_session, job_session)
                except Exception:
                    logger.exception(
                        "aisha_job_poller.poll_one_failed",
                        job_id=str(job.id),
                        gpu_session_id=str(job.gpu_session.id),
                    )

        await asyncio.gather(*(guarded(j) for j in jobs))

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
        ts = self._make_transition_service(session)
        product_id = str(job.product_id)

        # SSRF guard — validate tunnel hostname. Lowercase so Cloudflare
        # mixed-case responses don't bypass the suffix check.
        hostname = (gpu_session.tunnel_hostname or "").lower()
        try:
            validate_tunnel_hostname(hostname, allowed_suffix=self._allowed_tunnel_suffix)
        except InvalidTunnelHostnameError as exc:
            logger.error(
                "aisha_job_poller.invalid_tunnel_hostname",
                job_id=str(job.id),
                hostname=hostname,
                error=str(exc),
            )
            _, _ = await ts.transition_to_failed(
                job.id,
                error_message=f"Invalid tunnel hostname: {exc}"[:500],
                refund=True,
                product_id=product_id,
            )
            return

        prompt_id = job.external_request_id
        if not prompt_id:
            logger.warning(
                "aisha_job_poller.no_prompt_id",
                job_id=str(job.id),
            )
            return

        client = ComfyUIClient(
            f"https://{hostname}",
            request_timeout=self._config.comfyui_request_timeout_seconds,
        )
        try:
            await client.connect()
            history = await client.get_history(prompt_id)

            if prompt_id in history:
                await self._handle_history_complete(
                    client=client,
                    job=job,
                    history_entry=history[prompt_id],
                    product_id=product_id,
                    ts=ts,
                )
            else:
                queue = await client.get_queue()
                await self._handle_queue_state(
                    job=job,
                    queue=queue,
                    prompt_id=prompt_id,
                    product_id=product_id,
                    ts=ts,
                )
        except (TimeoutError, httpx.RequestError) as exc:
            # Transient — log and let the next tick try again. Age-based
            # timeout in _handle_queue_state will eventually mark FAILED.
            logger.warning(
                "aisha_job_poller.comfyui_transient_error",
                job_id=str(job.id),
                error_type=type(exc).__name__,
                error=str(exc),
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
        product_id: str,
        ts: JobStateTransitionService,
    ) -> None:
        if "outputs" not in history_entry:
            # History entry exists but no outputs — check for error flag.
            status_info = history_entry.get("status", {})
            if isinstance(status_info, dict) and status_info.get("status_str") == "error":
                messages = status_info.get("messages", [["Error", "Unknown"]])
                _, _ = await ts.transition_to_failed(
                    job.id,
                    error_message=f"ComfyUI reported error: {messages!r}"[:500],
                    refund=True,
                    product_id=product_id,
                )
                return

            # Neither outputs nor explicit error — apply same age-based timeout
            # as the queue path. Below threshold: log debug and let next tick retry.
            if self._is_job_past_timeout(job):
                logger.warning(
                    "aisha_job_poller.history_without_outputs_timeout",
                    job_id=str(job.id),
                    status=status_info,
                )
                _, _ = await ts.transition_to_failed(
                    job.id,
                    error_message=(
                        "ComfyUI history entry contained no outputs and no error; "
                        "job exceeded age timeout."
                    ),
                    refund=True,
                    product_id=product_id,
                )
            else:
                logger.debug(
                    "aisha_job_poller.history_without_outputs_transient",
                    job_id=str(job.id),
                    status=status_info,
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
        product_id: str,
        ts: JobStateTransitionService,
    ) -> None:
        running = queue.get("queue_running", [])
        is_running = any(item[1] == prompt_id for item in running if len(item) > 1)

        if is_running and job.status == JobStatus.QUEUED:
            await ts.transition_to_running(job.id)
            return

        # Absent from both history and queue — check age-based timeout.
        # Use started_at if set (job was observed running), fall back to
        # created_at so jobs that never reached RUNNING are still reaped.
        if self._is_job_past_timeout(job):
            _, _ = await ts.transition_to_failed(
                job.id,
                error_message="ComfyUI lost track of prompt — job timed out.",
                refund=True,
                product_id=product_id,
            )
            return

        # Below timeout threshold — emit warning if approaching it.
        reference = job.started_at or job.created_at
        if reference is not None:
            if reference.tzinfo is None:
                reference = reference.replace(tzinfo=UTC)
            age_seconds = (datetime.now(UTC) - reference).total_seconds()
            if age_seconds >= self._config.job_age_warning_seconds:
                logger.warning(
                    "aisha_job_poller.job_age_warning",
                    job_id=str(job.id),
                    age_seconds=int(age_seconds),
                )

        logger.debug(
            "aisha_job_poller.prompt_not_found_transient",
            job_id=str(job.id),
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _is_job_past_timeout(self, job: GenerationJob) -> bool:
        """True if the job is older than job_age_timeout_seconds.

        Uses started_at if set (job has been observed running), falling back to
        created_at so jobs that never reached RUNNING are still eventually reaped.
        Handles naive datetimes defensively by treating them as UTC.
        """
        reference = job.started_at or job.created_at
        if reference is None:
            return False
        if reference.tzinfo is None:
            reference = reference.replace(tzinfo=UTC)
        age = datetime.now(UTC) - reference
        return age.total_seconds() >= self._config.job_age_timeout_seconds

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

        ext, content_type = self._infer_image_format_and_content_type(filename)

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
    def _infer_image_format_and_content_type(filename: str) -> tuple[str, str]:
        """Return (format, content_type) inferred from filename. Defaults to png."""
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        return ext, _CONTENT_TYPES.get(ext, "image/png")

    @staticmethod
    def _collect_image_infos(history_entry: dict[str, Any]) -> list[dict[str, Any]]:
        images: list[dict[str, Any]] = []
        for _node_id, node_output in history_entry.get("outputs", {}).items():
            if "images" in node_output:
                images.extend(node_output["images"])
        return images
