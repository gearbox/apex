"""Aisha job service — DB-backed lifecycle for ComfyUI image jobs.

Polls ComfyUI for job completion, downloads outputs, uploads them to R2,
creates GenerationOutput records, and transitions GenerationJob status.

Used by UnifiedJobService in a poll-on-read pattern for QUEUED/RUNNING
Aisha image jobs (mirroring how GrokJobService handles Grok video jobs).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

import structlog

from src.api.services.comfyui_client import ComfyUIClient
from src.api.services.storage import R2StorageService, StorageType
from src.core.enums import JobStatus
from src.db.repositories.storage import StorageRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.storage import GenerationJob

logger = structlog.get_logger(__name__)

_CONTENT_TYPES: dict[str, str] = {
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "webp": "image/webp",
}


class AishaJobService:
    """DB-backed lifecycle service for Aisha/ComfyUI image jobs.

    Polls ComfyUI for job completion, downloads outputs, stores them
    in R2 and the database, and updates GenerationJob status.
    """

    def __init__(
        self,
        comfyui_client: ComfyUIClient,
        storage: R2StorageService,
        retention_days: int = 7,
    ) -> None:
        """Initialize the Aisha job service.

        Args:
            comfyui_client: ComfyUI HTTP client.
            storage: R2 storage service.
            retention_days: Days to retain generated content.
        """
        self._client = comfyui_client
        self._storage = storage
        self._retention_days = retention_days

    async def poll_image_job(
        self,
        session: AsyncSession,
        job_id: UUID,
    ) -> GenerationJob | None:
        """Poll ComfyUI for job status and persist completed results to DB.

        For QUEUED/RUNNING Aisha jobs this method:
        - Queries ComfyUI history; if completed, downloads each output image,
          uploads it to R2, creates a GenerationOutput DB record, and marks
          the job COMPLETED.
        - If the history entry signals an error, marks the job FAILED.
        - If the job is still in the live queue, transitions to RUNNING when
          appropriate.

        Idempotent: does nothing for jobs that are already terminal.

        Args:
            session: DB session (caller is responsible for commit/rollback).
            job_id: Job to poll.

        Returns:
            Updated GenerationJob if the status changed, else ``None``.
        """
        repo = StorageRepository(session)
        job = await repo.get_job(job_id)

        if job is None:
            return None
        if job.status in (JobStatus.COMPLETED.value, JobStatus.FAILED.value):
            return None

        prompt_id = job.external_request_id
        if not prompt_id:
            return None

        try:
            history = await self._client.get_history(prompt_id)
        except Exception:
            logger.warning("aisha_job.poll_history_failed", job_id=str(job_id))
            return None

        if prompt_id in history:
            return await self._handle_history_entry(session, repo, job, history[prompt_id])

        # Not yet in history — inspect the live queue
        try:
            queue = await self._client.get_queue()
        except Exception:
            logger.warning("aisha_job.poll_queue_failed", job_id=str(job_id))
            return None

        return await self._handle_queue_state(session, job, queue, prompt_id)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    async def _handle_history_entry(
        self,
        session: AsyncSession,
        repo: StorageRepository,
        job: GenerationJob,
        prompt_history: dict[str, Any],
    ) -> GenerationJob | None:
        """Process a ComfyUI history entry for the job."""
        if "outputs" in prompt_history:
            image_infos = self._collect_image_infos(prompt_history)
            stored = 0
            for idx, img_info in enumerate(image_infos):
                try:
                    await self._store_image(
                        session=session,
                        repo=repo,
                        job=job,
                        img_info=img_info,
                        output_index=idx,
                    )
                    stored += 1
                except Exception:
                    logger.warning(
                        "aisha_job.store_image_failed",
                        job_id=str(job.id),
                        output_index=idx,
                    )

            job.status = JobStatus.COMPLETED
            job.completed_at = datetime.now(UTC)
            await session.flush()
            logger.info("aisha_job.completed", job_id=str(job.id), output_count=stored)
            return job

        # History entry exists but carries no outputs — check for explicit error
        status_info = prompt_history.get("status", {})
        if status_info.get("status_str") == "error":
            messages = status_info.get("messages", [["Error", "Unknown"]])
            job.status = JobStatus.FAILED
            job.completed_at = datetime.now(UTC)
            job.error_message = str(messages)
            await session.flush()
            logger.warning("aisha_job.failed_by_comfyui", job_id=str(job.id))
            return job

        return None

    async def _handle_queue_state(
        self,
        session: AsyncSession,
        job: GenerationJob,
        queue: dict[str, Any],
        prompt_id: str,
    ) -> GenerationJob | None:
        """Transition job to RUNNING when it appears in the ComfyUI running queue."""
        running = queue.get("queue_running", [])
        is_running = any(item[1] == prompt_id for item in running if len(item) > 1)

        if is_running and job.status != JobStatus.RUNNING.value:
            job.status = JobStatus.RUNNING
            if job.started_at is None:
                job.started_at = datetime.now(UTC)
            await session.flush()
            return job

        return None

    def _collect_image_infos(self, history: dict[str, Any]) -> list[dict[str, Any]]:
        """Collect all image info dicts from a ComfyUI history entry."""
        images: list[dict[str, Any]] = []
        for _node_id, node_output in history.get("outputs", {}).items():
            if "images" in node_output:
                images.extend(node_output["images"])
        return images

    async def _store_image(
        self,
        *,
        session: AsyncSession,  # noqa: ARG002
        repo: StorageRepository,
        job: GenerationJob,
        img_info: dict[str, Any],
        output_index: int,
    ) -> None:
        """Download one ComfyUI output image and persist it in R2 + DB."""
        filename: str = img_info.get("filename", "")
        subfolder: str = img_info.get("subfolder", "")
        img_type: str = img_info.get("type", "output")
        if not filename:
            return

        data = await self._client.get_image(
            filename=filename, subfolder=subfolder, folder_type=img_type
        )

        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "png"
        content_type = _CONTENT_TYPES.get(ext, "image/png")

        result = await self._storage.upload(
            user_id=job.user_id,
            data=data,
            filename=filename,
            content_type=content_type,
            storage_type=StorageType.OUTPUT,
            job_id=job.id,
        )

        expires_at = datetime.now(UTC) + timedelta(days=self._retention_days)
        await repo.create_output(
            id=result.id,
            user_id=job.user_id,
            job_id=job.id,
            storage_key=result.storage_key,
            content_type=content_type,
            size_bytes=len(data),
            format=ext,
            output_index=output_index,
            expires_at=expires_at,
        )

        logger.debug(
            "aisha_job.output_stored",
            job_id=str(job.id),
            output_index=output_index,
            storage_key=result.storage_key,
        )
