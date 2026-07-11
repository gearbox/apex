"""Content retention sweep: enforces expires_at on outputs and uploads.

DB rows are the source of truth; deletion is DB-commit-first, then
best-effort R2 cleanup — identical rationale to
``ContentProxyService.delete_content()``: the worst case is a harmless
orphaned R2 object, never a dangling gallery reference.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog
from sqlalchemy import delete, exists, select, update

from src.api.services.storage.exceptions import StorageDeleteError
from src.db.models.storage import GenerationJob, GenerationOutput, UserImage
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

    from src.api.services.storage.r2 import R2StorageService

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class RetentionSweepResult:
    """Aggregate outcome of a single ``sweep()`` call (both content types)."""

    outputs_deleted: int
    uploads_deleted: int
    jobs_soft_deleted: int
    r2_keys_deleted: int
    r2_delete_failed: bool
    batches: int


@dataclass(frozen=True)
class _TypeSweepResult:
    """Per-content-type accumulator, merged into RetentionSweepResult by sweep()."""

    rows_deleted: int
    jobs_soft_deleted: int
    r2_keys_deleted: int
    r2_delete_failed: bool
    batches: int


class ContentRetentionService:
    """Pure sweep logic: drains expired outputs and uploads in bounded batches."""

    def __init__(
        self,
        *,
        session_factory: async_sessionmaker[AsyncSession],
        storage: R2StorageService,
        batch_size: int,
        max_batches_per_run: int,
    ) -> None:
        self._session_factory = session_factory
        self._storage = storage
        self._batch_size = batch_size
        self._max_batches_per_run = max_batches_per_run

    async def sweep(self) -> RetentionSweepResult:
        """Drain expired outputs then expired uploads, up to the per-type batch cap."""
        start = time.perf_counter()

        outputs_result = await self._sweep_outputs()
        uploads_result = await self._sweep_uploads()

        result = RetentionSweepResult(
            outputs_deleted=outputs_result.rows_deleted,
            uploads_deleted=uploads_result.rows_deleted,
            jobs_soft_deleted=outputs_result.jobs_soft_deleted,
            r2_keys_deleted=outputs_result.r2_keys_deleted + uploads_result.r2_keys_deleted,
            r2_delete_failed=outputs_result.r2_delete_failed or uploads_result.r2_delete_failed,
            batches=outputs_result.batches + uploads_result.batches,
        )

        duration_ms = int((time.perf_counter() - start) * 1000)
        logger.info(
            "content_retention.run",
            outputs_deleted=result.outputs_deleted,
            uploads_deleted=result.uploads_deleted,
            jobs_soft_deleted=result.jobs_soft_deleted,
            r2_keys_deleted=result.r2_keys_deleted,
            r2_delete_failed=result.r2_delete_failed,
            batches=result.batches,
            duration_ms=duration_ms,
        )
        return result

    async def _sweep_outputs(self) -> _TypeSweepResult:
        rows_deleted = 0
        jobs_soft_deleted = 0
        r2_keys_deleted = 0
        r2_delete_failed = False
        batches = 0

        for _ in range(self._max_batches_per_run):
            async with self._session_factory() as session:
                output_repo = OutputRepository(session)
                expired = await output_repo.get_expired(limit=self._batch_size)
                if not expired:
                    break
                batches += 1

                # Keys must be collected before the DELETE below — derivative
                # rows vanish via ON DELETE CASCADE and their keys are
                # unrecoverable afterward.
                expired_ids = [o.id for o in expired]
                derivative_map = await output_repo.batch_derivatives(expired_ids)
                keys = [d.storage_key for derivs in derivative_map.values() for d in derivs]
                keys.extend(o.storage_key for o in expired)

                touched_job_ids = {o.job_id for o in expired}

                delete_result = await session.execute(
                    delete(GenerationOutput).where(GenerationOutput.id.in_(expired_ids))
                )
                batch_rows_deleted = delete_result.rowcount or 0  # type: ignore[attr-defined]

                # Soft-delete only jobs touched by this batch that now have no
                # remaining outputs — never a table-wide scan. Note: this also
                # relies on generation_jobs.source_output_id / input_image_id
                # and generation_outputs.input_image_id being ON DELETE SET
                # NULL — sweeping nulls out lineage on downstream jobs, which
                # is expected (the gallery lineage schema treats these as
                # nullable already).
                soft_delete_result = await session.execute(
                    update(GenerationJob)
                    .where(
                        GenerationJob.id.in_(touched_job_ids),
                        GenerationJob.is_deleted.is_(False),
                        ~exists(
                            select(GenerationOutput.id).where(
                                GenerationOutput.job_id == GenerationJob.id
                            )
                        ),
                    )
                    .values(is_deleted=True)
                )
                batch_jobs_soft_deleted = soft_delete_result.rowcount or 0  # type: ignore[attr-defined]

                await session.commit()

            rows_deleted += batch_rows_deleted
            jobs_soft_deleted += batch_jobs_soft_deleted
            r2_keys_deleted += len(keys)

            if await self._delete_r2_keys(keys):
                r2_delete_failed = True

            logger.info(
                "content_retention.batch",
                content_type="output",
                rows_deleted=batch_rows_deleted,
                jobs_soft_deleted=batch_jobs_soft_deleted,
                r2_keys=len(keys),
            )

            if len(expired) < self._batch_size:
                break

        return _TypeSweepResult(
            rows_deleted=rows_deleted,
            jobs_soft_deleted=jobs_soft_deleted,
            r2_keys_deleted=r2_keys_deleted,
            r2_delete_failed=r2_delete_failed,
            batches=batches,
        )

    async def _sweep_uploads(self) -> _TypeSweepResult:
        rows_deleted = 0
        r2_keys_deleted = 0
        r2_delete_failed = False
        batches = 0

        for _ in range(self._max_batches_per_run):
            async with self._session_factory() as session:
                image_repo = UserImageRepository(session)
                expired = await image_repo.get_expired(limit=self._batch_size)
                if not expired:
                    break
                batches += 1

                expired_ids = [i.id for i in expired]
                derivative_map = await image_repo.batch_derivatives(expired_ids)
                keys = [d.storage_key for derivs in derivative_map.values() for d in derivs]
                keys.extend(i.storage_key for i in expired)

                # Derivative rows cascade via user_images.parent_image_id ON
                # DELETE CASCADE — no job linkage to soft-delete for uploads.
                delete_result = await session.execute(
                    delete(UserImage).where(UserImage.id.in_(expired_ids))
                )
                batch_rows_deleted = delete_result.rowcount or 0  # type: ignore[attr-defined]

                await session.commit()

            rows_deleted += batch_rows_deleted
            r2_keys_deleted += len(keys)

            if await self._delete_r2_keys(keys):
                r2_delete_failed = True

            logger.info(
                "content_retention.batch",
                content_type="upload",
                rows_deleted=batch_rows_deleted,
                jobs_soft_deleted=0,
                r2_keys=len(keys),
            )

            if len(expired) < self._batch_size:
                break

        return _TypeSweepResult(
            rows_deleted=rows_deleted,
            jobs_soft_deleted=0,
            r2_keys_deleted=r2_keys_deleted,
            r2_delete_failed=r2_delete_failed,
            batches=batches,
        )

    async def _delete_r2_keys(self, keys: list[str]) -> bool:
        """Best-effort R2 cleanup after the DB rows are already committed gone.

        Never raises — a storage-side failure must not undo or block the
        already-committed DB deletion (D3). Orphaned R2 objects are the
        accepted worst case.

        Returns:
            True if the delete failed (caller sets r2_delete_failed).
        """
        if not keys:
            return False
        try:
            await self._storage.delete_many(keys)
        except StorageDeleteError:
            logger.warning("content_retention.r2_delete_failed", key_count=len(keys))
            return True
        except Exception:
            logger.exception("content_retention.r2_delete_failed", key_count=len(keys))
            return True
        return False
