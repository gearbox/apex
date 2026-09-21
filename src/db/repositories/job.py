"""Repository for generation job database operations."""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import CursorResult, and_, exists, func, select, update
from sqlalchemy.orm import aliased, selectinload

from src.core.enums import GenerationType, GpuSessionStatus, JobStatus, Provider, TransactionType
from src.db.models.billing import TokenTransaction
from src.db.models.gpu_session import GpuSession
from src.db.models.storage import GenerationJob, GenerationOutput
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


@dataclasses.dataclass(frozen=True, slots=True)
class EmptyCompletion:
    """A completed Aisha job that has no output rows, with its debit if any."""

    job_id: UUID
    user_id: UUID
    product_id: str
    created_at: datetime
    account_id: UUID | None
    debit_amount: int | None


class JobRepository(BaseRepository[GenerationJob]):
    """Data access layer for GenerationJob records.

    Single-resource lookups accept an optional ``user_id`` kwarg.
    When provided the query includes a compound WHERE so ownership
    is enforced at the database level. When omitted (``None``),
    a plain primary-key lookup is used — suitable for internal /
    system operations such as background polling.
    """

    _model = GenerationJob

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    async def create(
        self,
        *,
        id: UUID,
        user_id: UUID,
        name: str,
        prompt: str,
        product_id: str,
        generation_type: GenerationType = GenerationType.I2I,
        status: JobStatus = JobStatus.PENDING,
        provider: Provider = Provider.GROK,
        model: str | None = None,
        aspect_ratio: str | None = None,
        source_job_id: UUID | None = None,
        source_output_id: UUID | None = None,
        primary_output_id: UUID | None = None,
        input_image_id: UUID | None = None,
        primary_upload_id: UUID | None = None,
        gpu_session_id: UUID | None = None,
    ) -> GenerationJob:
        """Create a new generation job.

        Args:
            id: Unique job ID.
            user_id: Owner of the job.
            name: Job name.
            prompt: Generation prompt.
            product_id: Product this job belongs to.
            generation_type: Type of generation (t2i, i2i, t2v, i2v).
            status: Initial status.
            provider: Generation provider (aisha, grok).
            model: Model identifier.
            aspect_ratio: Aspect ratio string, e.g. ``16:9``.
            source_job_id: ID of the source job for lineage tracking.
            source_output_id: ID of the source output used as input.
            primary_output_id: Generic source-media spelling for the primary
                output lineage. Mutually exclusive with ``source_output_id``.
            input_image_id: ID of the uploaded image used as input.
            gpu_session_id: GPU session this job will run on (Aisha only).

        Returns:
            Created GenerationJob instance.
        """
        if primary_output_id is not None:
            if source_output_id is not None and source_output_id != primary_output_id:
                raise ValueError("Conflicting primary output lineage values")
            source_output_id = primary_output_id
        if primary_upload_id is not None:
            if input_image_id is not None and input_image_id != primary_upload_id:
                raise ValueError("Conflicting primary upload lineage values")
            input_image_id = primary_upload_id

        job = GenerationJob(
            id=id,
            user_id=user_id,
            name=name,
            prompt=prompt,
            status=status,
            generation_type=generation_type,
            provider=provider,
            model=model,
            aspect_ratio=aspect_ratio,
            product_id=product_id,
            source_job_id=source_job_id,
            source_output_id=source_output_id,
            input_image_id=input_image_id,
            gpu_session_id=gpu_session_id,
        )
        self._session.add(job)
        await self._session.flush()
        return job

    async def get(
        self,
        job_id: UUID,
        *,
        user_id: UUID | None = None,
    ) -> GenerationJob | None:
        """Get a job by ID, optionally scoped to a user.

        When ``user_id`` is provided, soft-deleted jobs are excluded
        (user-facing). When ``user_id`` is ``None``, all jobs are
        returned including soft-deleted (internal/system use).

        Args:
            job_id: Job ID to look up.
            user_id: When provided, only returns if owned by this user
                and not soft-deleted. ``None`` skips both checks
                (for internal use).

        Returns:
            GenerationJob if found, None otherwise.
        """
        if user_id is None:
            return await self._session.get(GenerationJob, job_id)

        result = await self._session.execute(
            select(GenerationJob).where(
                GenerationJob.id == job_id,
                GenerationJob.user_id == user_id,
                GenerationJob.is_deleted.is_(False),
            )
        )
        return result.scalar_one_or_none()

    async def get_many(
        self,
        ids: Sequence[UUID],
        *,
        user_id: UUID,
    ) -> dict[UUID, GenerationJob]:
        """Batch-fetch jobs by ID, ownership-scoped (soft-deleted excluded).

        Args:
            ids: Job IDs to look up.
            user_id: Owner — rows of other users are excluded.

        Returns:
            Mapping from id to GenerationJob for rows that exist, are owned
            by ``user_id``, and are not soft-deleted. Missing/foreign ids
            are simply absent from the result.
        """
        if not ids:
            return {}

        result = await self._session.execute(
            select(GenerationJob).where(
                GenerationJob.id.in_(ids),
                GenerationJob.user_id == user_id,
                GenerationJob.is_deleted.is_(False),
            )
        )
        return {job.id: job for job in result.scalars().all()}

    async def update_status(
        self,
        job_id: UUID,
        status: JobStatus,
        *,
        external_request_id: str | None = None,
        started_at: datetime | None = None,
        completed_at: datetime | None = None,
    ) -> GenerationJob | None:
        """Update job status and timestamps.

        Args:
            job_id: Job ID to update.
            status: New status.
            external_request_id: External provider request ID.
            started_at: Job start time (optional).
            completed_at: Job completion time (optional).

        Returns:
            Updated GenerationJob if found, None otherwise.
        """
        job = await self.get(job_id)
        if job is None:
            return None

        job.status = status
        if external_request_id is not None:
            job.external_request_id = external_request_id
        if started_at is not None:
            job.started_at = started_at
        if completed_at is not None:
            job.completed_at = completed_at

        await self._session.flush()
        return job

    async def list_by_user(
        self,
        user_id: UUID,
        *,
        status: JobStatus | None = None,
        provider: Provider | None = None,
        generation_type: GenerationType | None = None,
        limit: int = 20,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
        eager_load_outputs: bool = False,
    ) -> Sequence[GenerationJob]:
        """List jobs for a user with cursor-based pagination and optional filters.

        Uses limit+1 fetch pattern. Caller checks ``len(result) > limit``
        to determine ``has_more``.

        Args:
            user_id: User to list jobs for.
            status: Filter by job status (optional).
            provider: Filter by provider (optional).
            generation_type: Filter by generation type (optional).
            limit: Page size (fetches limit+1 for has_more check).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.
            eager_load_outputs: When True, eagerly loads the ``outputs``
                relationship via ``selectinload`` to avoid N+1 queries
                when building response DTOs.

        Returns:
            List of GenerationJob instances ordered by
            ``(created_at DESC, id DESC)``.
        """
        from sqlalchemy import literal, tuple_

        query = select(GenerationJob).where(GenerationJob.user_id == user_id)

        # Exclude soft-deleted jobs from user-facing listings
        query = query.where(GenerationJob.is_deleted.is_(False))

        if status is not None:
            query = query.where(GenerationJob.status == status)
        if provider is not None:
            query = query.where(GenerationJob.provider == provider)
        if generation_type is not None:
            query = query.where(GenerationJob.generation_type == generation_type)

        if eager_load_outputs:
            query = query.options(selectinload(GenerationJob.outputs))

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(GenerationJob.created_at, GenerationJob.id)
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                GenerationJob.created_at.desc(),
                GenerationJob.id.desc(),
            ).limit(limit + 1)
        )
        return result.scalars().all()

    async def soft_delete(
        self,
        job_id: UUID,
        *,
        user_id: UUID,
    ) -> GenerationJob | None:
        """Soft-delete a job — marks it as deleted without removing the record.

        The job record and R2 outputs are retained until the retention
        policy cleans them up. The job stops appearing in user-facing
        list/get results.

        Uses a direct query that bypasses the is_deleted filter so that
        soft-deleting an already-deleted job is idempotent.

        Args:
            job_id: Job to soft-delete.
            user_id: Owner check — only the owner can delete their own job.

        Returns:
            Updated GenerationJob if found and owned, None otherwise.
        """
        result = await self._session.execute(
            select(GenerationJob).where(
                GenerationJob.id == job_id,
                GenerationJob.user_id == user_id,
            )
        )
        job = result.scalar_one_or_none()
        if job is None:
            return None

        job.is_deleted = True
        await self._session.flush()
        return job

    async def list_aisha_jobs_for_polling(
        self,
        *,
        limit: int = 200,
    ) -> Sequence[GenerationJob]:
        """Return in-progress Aisha jobs whose GPU session is active.

        Eager-loads ``gpu_session`` so the poller can read tunnel_hostname
        on detached objects without triggering additional SQL.

        Args:
            limit: Maximum jobs to return per tick.

        Returns:
            GenerationJob instances ordered by ``created_at ASC`` (oldest first
            so long-running jobs are not starved).
        """
        result = await self._session.execute(
            select(GenerationJob)
            .join(GpuSession, GenerationJob.gpu_session_id == GpuSession.id)
            .where(
                GenerationJob.provider == Provider.AISHA,
                GenerationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                GpuSession.status == GpuSessionStatus.active,
                GenerationJob.is_deleted.is_(False),
            )
            .options(selectinload(GenerationJob.gpu_session))
            .order_by(GenerationJob.created_at.asc())
            .limit(limit)
        )
        return result.scalars().all()

    async def list_in_flight_for_session(
        self,
        gpu_session_id: UUID,
    ) -> Sequence[GenerationJob]:
        """Return Aisha jobs that are QUEUED/RUNNING for one GPU session.

        Used by the session-termination sweep to identify jobs that need to be
        transitioned to FAILED and refunded.

        No JOIN on GpuSession.status — the caller is in a transactional context
        where the session's status is being changed concurrently; filtering on
        GpuSession.status here would create a TOCTOU window.

        Args:
            gpu_session_id: Session whose in-flight jobs to list.

        Returns:
            GenerationJob instances (no eager-loading needed; the sweep only
            reads the id).
        """
        result = await self._session.execute(
            select(GenerationJob).where(
                GenerationJob.gpu_session_id == gpu_session_id,
                GenerationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                GenerationJob.is_deleted.is_(False),
            )
        )
        return result.scalars().all()

    async def count_in_flight_for_session(
        self,
        gpu_session_id: UUID,
    ) -> int:
        """Count Aisha jobs that are QUEUED/RUNNING for one GPU session.

        Used by:
        - pause_session precondition (reject if non-zero)
        - GET /v1/sessions/{id} response (in_flight_job_count field for
          frontend Pause-button gating)

        Args:
            gpu_session_id: Session whose in-flight jobs to count.

        Returns:
            Count of QUEUED/RUNNING non-deleted jobs for the session.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(GenerationJob)
            .where(
                GenerationJob.gpu_session_id == gpu_session_id,
                GenerationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                GenerationJob.is_deleted.is_(False),
            )
        )
        return int(result.scalar_one())

    async def count_in_flight_for_session_and_model(
        self,
        gpu_session_id: UUID,
        model_type: str,
    ) -> int:
        """Count Aisha jobs that are QUEUED/RUNNING for one GPU session and model type.

        Used by the P4 remove endpoint (D37): removal is blocked by an in-flight
        job of the model type being removed, but not by jobs on another model
        sharing the same session.

        Args:
            gpu_session_id: Session whose in-flight jobs to count.
            model_type: ModelType value to filter on.

        Returns:
            Count of QUEUED/RUNNING non-deleted jobs for the session+model pair.
        """
        result = await self._session.execute(
            select(func.count())
            .select_from(GenerationJob)
            .where(
                GenerationJob.gpu_session_id == gpu_session_id,
                GenerationJob.model == model_type,
                GenerationJob.status.in_([JobStatus.QUEUED, JobStatus.RUNNING]),
                GenerationJob.is_deleted.is_(False),
            )
        )
        return int(result.scalar_one())

    async def list_pending_video_jobs(
        self,
        provider: Provider = Provider.GROK,
    ) -> Sequence[GenerationJob]:
        """List pending video generation jobs for polling.

        Returns jobs that are queued or running, have a video generation
        type (T2V, I2V, or V2V — Grok's ``grok-imagine-video`` supports all
        three), have an ``external_request_id`` set (indicating the provider
        accepted the request), and are not soft-deleted.

        Args:
            provider: Provider enum to filter by.

        Returns:
            List of GenerationJob instances needing polling.
        """
        video_types = [GenerationType.T2V, GenerationType.I2V, GenerationType.V2V]
        pending_statuses = [JobStatus.QUEUED, JobStatus.RUNNING]

        result = await self._session.execute(
            select(GenerationJob)
            .where(GenerationJob.provider == provider)
            .where(GenerationJob.status.in_(pending_statuses))
            .where(GenerationJob.generation_type.in_(video_types))
            .where(GenerationJob.external_request_id.isnot(None))
            .where(GenerationJob.is_deleted.is_(False))
        )
        return result.scalars().all()

    async def list_empty_completions(
        self,
        *,
        product_id: str | None = None,
        since: datetime | None = None,
        job_ids: Sequence[UUID] | None = None,
        limit: int | None = None,
    ) -> Sequence[EmptyCompletion]:
        """List Aisha jobs marked ``completed`` that have no ``generation_outputs`` row.

        Used by the empty-completion refund backfill. ``is_deleted`` jobs are
        excluded: the retention sweeper deletes expired output rows and
        soft-deletes the job, so a soft-deleted completed job with no outputs is
        the normal end of a job's life, not a defect.

        Note a job whose owner later deleted all of its outputs is also
        ``completed`` with no outputs and is *not* soft-deleted — the database
        cannot tell it from a defective completion, so callers must let a human
        review the result (or pass ``job_ids``) before acting on it.

        Args:
            product_id: Restrict to one product.
            since: Restrict to jobs created at or after this instant.
            job_ids: Restrict to exactly these jobs (still subject to every
                other predicate).
            limit: Maximum rows to return.

        Returns:
            Rows ordered by ``created_at ASC, id ASC``. ``account_id`` and
            ``debit_amount`` (absolute tokens) are ``None`` for a job with no
            debit ledger row.
        """
        debit = aliased(TokenTransaction)
        stmt = (
            select(
                GenerationJob.id,
                GenerationJob.user_id,
                GenerationJob.product_id,
                GenerationJob.created_at,
                debit.account_id,
                func.abs(debit.amount),
            )
            .outerjoin(
                debit,
                and_(
                    debit.job_id == GenerationJob.id,
                    debit.transaction_type == TransactionType.DEBIT.value,
                ),
            )
            .where(
                GenerationJob.provider == Provider.AISHA,
                GenerationJob.status == JobStatus.COMPLETED,
                GenerationJob.is_deleted.is_(False),
                ~exists(
                    select(GenerationOutput.id).where(GenerationOutput.job_id == GenerationJob.id)
                ),
            )
            .order_by(GenerationJob.created_at.asc(), GenerationJob.id.asc())
        )
        if product_id is not None:
            stmt = stmt.where(GenerationJob.product_id == product_id)
        if since is not None:
            stmt = stmt.where(GenerationJob.created_at >= since)
        if job_ids is not None:
            stmt = stmt.where(GenerationJob.id.in_(job_ids))
        if limit is not None:
            stmt = stmt.limit(limit)

        result = await self._session.execute(stmt)
        return [
            EmptyCompletion(
                job_id=row[0],
                user_id=row[1],
                product_id=row[2],
                created_at=row[3],
                account_id=row[4],
                debit_amount=int(row[5]) if row[5] is not None else None,
            )
            for row in result.all()
        ]

    async def mark_empty_completion_failed(
        self,
        job_id: UUID,
        *,
        failure_code: str,
        error_message: str,
        public_error_message: str | None = None,
    ) -> bool:
        """COMPLETED → FAILED, only when the job has zero outputs. Returns whether it changed.

        A dedicated method rather than a general ``COMPLETED → FAILED``
        transition (D6): the ``WHERE status = 'completed' AND NOT EXISTS
        (outputs)`` predicate is the guard, so this cannot be pointed at a job
        that has a result. Deliberately not on ``JobStateTransitionService``,
        whose terminal states stay terminal for every other caller.

        Does not commit and does not touch billing — the caller pairs it with
        ``BillingService.refund`` in one transaction.

        Args:
            job_id: Job to correct.
            failure_code: Stable machine-readable failure category.
            error_message: Internal diagnostic. Never shown to users.
            public_error_message: User-facing text; when omitted the row falls
                back to the fixed legacy failed-job message.

        Returns:
            True if this call changed the row; False if the job is not
            ``completed`` or has outputs (row untouched).
        """
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GenerationJob)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.status == JobStatus.COMPLETED,
                    ~exists(select(GenerationOutput.id).where(GenerationOutput.job_id == job_id)),
                )
                .values(
                    status=JobStatus.FAILED,
                    failure_code=failure_code[:100],
                    error_message=error_message[:2000],
                    public_error_message=(
                        public_error_message[:2000] if public_error_message is not None else None
                    ),
                )
                .execution_options(synchronize_session=False)
            ),
        )
        return result.rowcount == 1
