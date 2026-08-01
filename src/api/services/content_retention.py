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
from sqlalchemy import delete, exists, func, or_, select, update

from src.api.services.storage.exceptions import StorageDeleteError
from src.core.library_ref import LibraryAssetSource
from src.db.models.library import LibraryAssetMetadata
from src.db.models.storage import (
    GenerationJob,
    GenerationMaterializationAttempt,
    GenerationOutput,
    StorageCleanupRecord,
    UserImage,
)
from src.db.repositories.library_tag import LibraryTagRepository
from src.db.repositories.output import OutputRepository
from src.db.repositories.user_image import UserImageRepository

if TYPE_CHECKING:
    from uuid import UUID

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

    async def reconcile_storage_artifacts(self) -> tuple[int, int]:
        """Run restart-safe materialization and migration-outbox cleanup.

        Kept separate from retention's normal sweep so media expiry work has
        stable bounded behavior; ``ContentRetentionWorker`` invokes both on
        its periodic tick.
        """
        reconciled_attempts = await self._reconcile_materialization_attempts()
        cleanup_records = await self._drain_storage_cleanup_outbox()
        logger.info(
            "content_retention.storage_reconciliation",
            materialization_attempts_reconciled=reconciled_attempts,
            storage_cleanup_records_drained=cleanup_records,
        )
        return reconciled_attempts, cleanup_records

    async def _reconcile_materialization_attempts(self) -> int:
        """Clean abandoned video attempts without invalidating a live owner.

        Lease expiry alone is not abandonment: the original owner may still
        complete while its token remains stored.  An attempt becomes eligible
        only after a terminal job result or token replacement/clearance.
        """
        async with self._session_factory() as session:
            result = await session.execute(
                select(GenerationMaterializationAttempt, GenerationJob.status)
                .join(GenerationJob, GenerationJob.id == GenerationMaterializationAttempt.job_id)
                .where(
                    GenerationMaterializationAttempt.state.in_(
                        ["planned", "uploading", "cleanup_pending", "ambiguous"]
                    ),
                    or_(
                        GenerationJob.status.in_(["completed", "failed", "cancelled"]),
                        GenerationJob.finalization_claim_token.is_(None),
                        GenerationJob.finalization_claim_token
                        != GenerationMaterializationAttempt.claim_token,
                    ),
                )
                .order_by(GenerationMaterializationAttempt.updated_at)
                .limit(self._batch_size)
            )
            candidates = list(result.all())

        reconciled = 0
        for attempt, _status in candidates:
            async with self._session_factory() as session:
                referenced_keys = set(
                    (
                        await session.scalars(
                            select(GenerationOutput.storage_key).where(
                                GenerationOutput.job_id == attempt.job_id
                            )
                        )
                    ).all()
                )
                unreferenced = [
                    key for key in attempt.planned_storage_keys if key not in referenced_keys
                ]
            try:
                if unreferenced:
                    await self._storage.delete_many(unreferenced)
            except Exception as exc:
                async with self._session_factory() as session:
                    await session.execute(
                        update(GenerationMaterializationAttempt)
                        .where(GenerationMaterializationAttempt.id == attempt.id)
                        .values(
                            state="cleanup_pending",
                            reconciliation_error=str(exc)[:2000],
                        )
                    )
                    await session.commit()
                logger.exception(
                    "content_retention.materialization_cleanup_failed",
                    attempt_id=str(attempt.id),
                )
                continue

            async with self._session_factory() as session:
                await session.execute(
                    update(GenerationMaterializationAttempt)
                    .where(GenerationMaterializationAttempt.id == attempt.id)
                    .values(
                        state="cleaned" if unreferenced else "committed",
                        reconciliation_error=None,
                    )
                )
                await session.commit()
            reconciled += 1
        return reconciled

    async def _drain_storage_cleanup_outbox(self) -> int:
        """Retry post-commit R2 cleanup queued by data migrations and services."""
        async with self._session_factory() as session:
            records = list(
                (
                    await session.scalars(
                        select(StorageCleanupRecord)
                        .where(StorageCleanupRecord.state == "pending")
                        .order_by(StorageCleanupRecord.created_at)
                        .limit(self._batch_size)
                    )
                ).all()
            )

        deleted = 0
        for record in records:
            try:
                await self._storage.delete(record.storage_key)
            except Exception as exc:
                async with self._session_factory() as session:
                    await session.execute(
                        update(StorageCleanupRecord)
                        .where(StorageCleanupRecord.id == record.id)
                        .values(
                            attempts=StorageCleanupRecord.attempts + 1,
                            last_error=str(exc)[:2000],
                        )
                    )
                    await session.commit()
                logger.exception(
                    "content_retention.storage_cleanup_outbox_failed",
                    cleanup_record_id=str(record.id),
                )
                continue

            async with self._session_factory() as session:
                await session.execute(
                    update(StorageCleanupRecord)
                    .where(StorageCleanupRecord.id == record.id)
                    .values(state="deleted", processed_at=func.now(), last_error=None)
                )
                await session.commit()
            deleted += 1
        return deleted

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

                metadata_purged = await self._purge_metadata(
                    session, LibraryAssetSource.OUTPUT, expired_ids
                )
                tags_purged = await self._purge_tags(
                    session, LibraryAssetSource.OUTPUT, expired_ids
                )

                await session.commit()

            rows_deleted += batch_rows_deleted
            jobs_soft_deleted += batch_jobs_soft_deleted
            logger.info("content_retention.metadata_purged", count=metadata_purged)
            logger.info("content_retention.tags_purged", count=tags_purged)

            deleted, failed = await self._delete_r2_keys(keys)
            r2_keys_deleted += deleted
            if failed:
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

                metadata_purged = await self._purge_metadata(
                    session, LibraryAssetSource.UPLOAD, expired_ids
                )
                tags_purged = await self._purge_tags(
                    session, LibraryAssetSource.UPLOAD, expired_ids
                )

                await session.commit()

            rows_deleted += batch_rows_deleted
            logger.info("content_retention.metadata_purged", count=metadata_purged)
            logger.info("content_retention.tags_purged", count=tags_purged)

            deleted, failed = await self._delete_r2_keys(keys)
            r2_keys_deleted += deleted
            if failed:
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

    async def _purge_metadata(
        self,
        session: AsyncSession,
        source: LibraryAssetSource,
        asset_ids: list[UUID],
    ) -> int:
        """Purge library_asset_metadata rows for a just-deleted batch (D14).

        Runs in the same transaction as the batch's row DELETE, before that
        transaction's commit — a polymorphic FK is impossible here (asset_type
        discriminates two different tables), so this explicit purge is the
        only way to keep library_asset_metadata from accumulating orphans.
        """
        if not asset_ids:
            return 0
        result = await session.execute(
            delete(LibraryAssetMetadata).where(
                LibraryAssetMetadata.asset_type == source.value,
                LibraryAssetMetadata.asset_id.in_(asset_ids),
            )
        )
        return result.rowcount or 0  # type: ignore[attr-defined]

    async def _purge_tags(
        self,
        session: AsyncSession,
        source: LibraryAssetSource,
        asset_ids: list[UUID],
    ) -> int:
        """Purge library_asset_tags rows for a just-deleted batch.

        Same rationale and same-transaction placement as ``_purge_metadata``
        — a polymorphic FK is impossible here, so this explicit purge is the
        only way to keep library_asset_tags from accumulating orphans.
        """
        if not asset_ids:
            return 0
        return await LibraryTagRepository(session).delete_tags_for_assets(
            [(source.value, asset_id) for asset_id in asset_ids]
        )

    async def _delete_r2_keys(self, keys: list[str]) -> tuple[int, bool]:
        """Best-effort R2 cleanup after the DB rows are already committed gone.

        Returns (deleted_count, failed). Never raises (D3): a storage-side
        failure must not undo the committed DB deletion. Note the ordering
        consequence — once the DB rows are gone, the sweeper can never revisit
        these keys, so a transient R2 outage permanently orphans this batch's
        objects. That is the accepted D3 trade-off; an R2 bucket lifecycle rule
        remains a deferred defense-in-depth for such orphans.
        """
        if not keys:
            return 0, False
        try:
            deleted = await self._storage.delete_many(keys)
        except StorageDeleteError:
            logger.warning("content_retention.r2_delete_failed", key_count=len(keys))
            return 0, True
        except Exception:
            logger.exception("content_retention.r2_delete_failed", key_count=len(keys))
            return 0, True
        return deleted, False
