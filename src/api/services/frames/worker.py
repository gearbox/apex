"""FrameExtractionWorker — claims and executes video frame extraction jobs.

Lifecycle clone of GrokVideoWorker (start/stop/_run_loop via PeriodicWorker).
Per tick: claim one queued job (FOR UPDATE SKIP LOCKED, commit immediately —
see pitfall #6 in the implementation prompt: never hold the row lock across
the ffmpeg work), do the actual extraction against a downloaded temp file,
then a second short transaction persists the result. One job's failure never
kills the loop — the base PeriodicWorker already logs+swallows run_once()
exceptions, but every failure here is caught locally so it can be recorded
on the job row (``status=failed``, ``error=<real message>``) rather than
just logged.
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

from src.api.services.frames import ffmpeg as frame_ffmpeg
from src.api.services.image_thumbnail import make_image_thumbnails, read_dimensions
from src.api.services.storage import MediaFormat, StorageType
from src.core.enums import FrameExtractionKind, MediaKind, media_kind_from_content_type
from src.db.repositories.frame_extraction import FrameExtractionJobRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository
from src.workers.base import PeriodicWorker

if TYPE_CHECKING:
    from collections.abc import Callable
    from uuid import UUID

    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.storage.r2 import R2StorageService
    from src.core.config import Settings
    from src.db import DatabaseManager
    from src.db.models.frame_extraction import FrameExtractionJob

logger = structlog.get_logger(__name__)

# Top-level prefix (not users/{uid}/...) — R2 lifecycle rules filter on a
# literal prefix, so a per-user path fragment can't be covered by one rule.
# Ownership is enforced by job-row ownership in FrameExtractionService, not
# by key prefix. Requires an out-of-band R2 lifecycle rule expiring this
# prefix after settings.frame_preview_retention_days (see Settings docstring).
_PREVIEW_KEY_PREFIX = "frame-previews"


class _SourceNotFoundError(Exception):
    """Internal: the job's source row is missing despite the job existing.

    Should be unreachable — frame_extraction_jobs cascades away with its
    source (ON DELETE CASCADE) — but guards against a race between claim and
    source lookup within the same transaction.
    """


class FrameExtractionWorker(PeriodicWorker):
    """Polls frame_extraction_jobs for queued work and executes it."""

    def __init__(
        self,
        db_manager: DatabaseManager,
        r2_storage: R2StorageService,
        settings: Settings,
        *,
        redis_enabled: bool = False,
        redis_client_factory: Callable[[], Redis],
    ) -> None:
        super().__init__(
            name="frame_extraction",
            interval_seconds=settings.frame_extract_poll_interval_seconds,
            redis_enabled=redis_enabled,
            redis_client_factory=redis_client_factory,
        )
        self._db_manager = db_manager
        self._storage = r2_storage
        self._ffmpeg_timeout = settings.frame_extract_ffmpeg_timeout_seconds
        self._preview_max_edge = settings.frame_preview_max_edge
        self._retention_days = settings.retention_days
        self._stale_running_seconds = settings.frame_extract_stale_running_seconds

    async def run_once(self) -> None:
        """Fail stale running jobs, then claim (and commit) one job, then
        execute it outside any DB transaction."""
        async with self._db_manager.session() as session:
            cutoff = datetime.now(UTC) - timedelta(seconds=self._stale_running_seconds)
            stale_count = await FrameExtractionJobRepository(session).fail_stale_running(
                cutoff=cutoff
            )
            await session.commit()
            if stale_count > 0:
                logger.warning("frames.stale_jobs_failed", count=stale_count)

        async with self._db_manager.session() as session:
            job = await FrameExtractionJobRepository(session).claim_next()
            if job is None:
                return

            try:
                storage_key, content_type = await self._load_source(session, job)
            except _SourceNotFoundError as e:
                # Shouldn't happen (cascade deletes the job with its source),
                # but fail the job rather than crash the worker loop.
                await FrameExtractionJobRepository(session).mark_failed(job.id, error=str(e))
                await session.commit()
                logger.exception("frames.job_failed", job_id=str(job.id))
                return

            await session.commit()

        logger.info("frames.job_claimed", job_id=str(job.id), kind=job.kind)

        try:
            result = await self._execute(job, storage_key, content_type)
        except Exception as e:
            logger.exception("frames.job_failed", job_id=str(job.id))
            async with self._db_manager.session() as session:
                await FrameExtractionJobRepository(session).mark_failed(job.id, error=str(e))
                await session.commit()
            return

        async with self._db_manager.session() as session:
            await FrameExtractionJobRepository(session).mark_completed(job.id, result=result)
            await session.commit()
        logger.info("frames.job_completed", job_id=str(job.id), kind=job.kind)

    async def _load_source(self, session: AsyncSession, job: FrameExtractionJob) -> tuple[str, str]:
        if job.source_output_id is not None:
            output = await OutputRepository(session).get(job.source_output_id)
            if output is None:
                raise _SourceNotFoundError(f"Source output {job.source_output_id} not found")
            return output.storage_key, output.content_type

        if job.source_upload_id is not None:
            upload = await UserImageRepository(session).get(job.source_upload_id)
            if upload is None:
                raise _SourceNotFoundError(f"Source upload {job.source_upload_id} not found")
            return upload.storage_key, upload.content_type

        raise _SourceNotFoundError(f"Job {job.id} has no source set")

    async def _execute(
        self, job: FrameExtractionJob, storage_key: str, content_type: str
    ) -> dict[str, object]:
        """Download the source, probe it, extract frames. Raises on any failure."""
        if media_kind_from_content_type(content_type) is not MediaKind.VIDEO:
            # Defense-in-depth: FrameExtractionService already rejects
            # non-video sources at job creation time.
            raise ValueError(f"Source content type {content_type!r} is not a video")

        video_bytes = await self._storage.download(storage_key)
        video_path = await asyncio.to_thread(self._write_temp, video_bytes)
        try:
            probe = await frame_ffmpeg.probe(video_path, timeout_seconds=self._ffmpeg_timeout)

            if job.kind == FrameExtractionKind.PREVIEW.value:
                return await self._run_preview(job, video_path, probe.duration_ms)
            if job.kind == FrameExtractionKind.EXTRACT.value:
                return await self._run_extract(job, video_path, probe.duration_ms)
            raise ValueError(f"Unknown frame extraction kind: {job.kind!r}")
        finally:
            await asyncio.to_thread(video_path.unlink, missing_ok=True)

    @staticmethod
    def _write_temp(data: bytes) -> Path:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            f.write(data)
            return Path(f.name)

    # -------------------------------------------------------------------
    # Preview
    # -------------------------------------------------------------------

    async def _run_preview(
        self, job: FrameExtractionJob, video_path: Path, duration_ms: int
    ) -> dict[str, object]:
        frame_count = job.params["frame_count"]
        timestamps = frame_ffmpeg.compute_uniform_timestamps(duration_ms, frame_count)

        frames_out: list[dict[str, object]] = []
        for index, ts in enumerate(timestamps):
            webp_bytes = await frame_ffmpeg.extract_frame(
                video_path,
                ts,
                out_format="webp",
                max_edge=self._preview_max_edge,
                timeout_seconds=self._ffmpeg_timeout,
            )
            key = f"{_PREVIEW_KEY_PREFIX}/{job.user_id}/{job.id}/{index:03d}.webp"
            await self._storage.put_raw(key, webp_bytes, content_type="image/webp")
            frames_out.append({"index": index, "timestamp_ms": ts, "key": key})

        return {"frames": frames_out, "duration_ms": duration_ms}

    # -------------------------------------------------------------------
    # Extract
    # -------------------------------------------------------------------

    async def _run_extract(
        self, job: FrameExtractionJob, video_path: Path, duration_ms: int
    ) -> dict[str, object]:
        timestamps: list[int] = job.params["timestamps_ms"]
        if out_of_range := [ts for ts in timestamps if ts < 0 or ts >= duration_ms]:
            raise ValueError(f"Timestamps out of range [0, {duration_ms})ms: {out_of_range}")

        expires_at = datetime.now(UTC) + timedelta(days=self._retention_days)
        frames_out: list[dict[str, object]] = []
        uploaded_keys: list[str] = []

        try:
            async with self._db_manager.session() as session:
                image_repo = UserImageRepository(session)
                for ts in timestamps:
                    upload_id = await self._extract_and_save_frame(
                        job,
                        video_path,
                        ts,
                        image_repo=image_repo,
                        expires_at=expires_at,
                        uploaded_keys=uploaded_keys,
                    )
                    frames_out.append({"timestamp_ms": ts, "upload_id": str(upload_id)})
                await session.commit()
        except Exception:
            if uploaded_keys:
                try:
                    await self._storage.delete_many(uploaded_keys)
                except Exception:
                    logger.warning(
                        "frames.extract_orphan_cleanup_failed",
                        job_id=str(job.id),
                        key_count=len(uploaded_keys),
                    )
            raise

        return {"frames": frames_out}

    async def _extract_and_save_frame(
        self,
        job: FrameExtractionJob,
        video_path: Path,
        timestamp_ms: int,
        *,
        image_repo: UserImageRepository,
        expires_at: datetime,
        uploaded_keys: list[str],
    ) -> UUID:
        png_bytes = await frame_ffmpeg.extract_frame(
            video_path,
            timestamp_ms,
            out_format="png",
            timeout_seconds=self._ffmpeg_timeout,
        )
        upload_result = await self._storage.upload(
            user_id=job.user_id,
            data=png_bytes,
            content_type=MediaFormat.PNG.content_type,
            storage_type=StorageType.UPLOAD,
        )
        uploaded_keys.append(upload_result.storage_key)

        dims = await read_dimensions(png_bytes)

        db_image = await image_repo.create(
            id=upload_result.id,
            user_id=job.user_id,
            storage_key=upload_result.storage_key,
            original_filename=f"{upload_result.id}.png",
            content_type=MediaFormat.PNG.content_type,
            size_bytes=len(png_bytes),
            format=MediaFormat.PNG.value,
            expires_at=expires_at,
            product_id=job.product_id,
            width=dims.width if dims is not None else None,
            height=dims.height if dims is not None else None,
            source_output_id=job.source_output_id,
            source_upload_id=job.source_upload_id,
            source_timestamp_ms=timestamp_ms,
        )

        # WEBP sm/md derivatives — non-fatal, mirrors UserContentService.upload_image.
        try:
            thumbnails = await make_image_thumbnails(png_bytes)
            for generated in thumbnails:
                thumb_result = await self._storage.upload(
                    user_id=job.user_id,
                    data=generated.result.data,
                    content_type=generated.result.content_type,
                    storage_type=StorageType.UPLOAD,
                )
                uploaded_keys.append(thumb_result.storage_key)
                await image_repo.create(
                    id=thumb_result.id,
                    user_id=job.user_id,
                    storage_key=thumb_result.storage_key,
                    original_filename=f"{thumb_result.id}.{generated.result.format}",
                    content_type=generated.result.content_type,
                    size_bytes=len(generated.result.data),
                    format=generated.result.format,
                    expires_at=expires_at,
                    product_id=job.product_id,
                    is_thumbnail=True,
                    parent_image_id=db_image.id,
                    thumbnail_max_edge=generated.spec.max_edge,
                    width=generated.result.width,
                    height=generated.result.height,
                )
        except Exception:
            logger.warning("frames.extract_thumbnail_generation_failed", image_id=str(db_image.id))

        return db_image.id
