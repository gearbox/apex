"""Provider-neutral materialization of a completed remote video.

Owns the crash-safe completion path shared by every async-video provider:
the PostgreSQL finalization claim, the download/ingest/R2 upload, the durable
materialization attempt, and the reconciliation of uncertain outcomes.
Everything provider-specific (log origin, error settlement, retention, lease
length) is injected at construction — nothing here reads ``Settings``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING
from uuid import uuid4

import httpx
import structlog
from sqlalchemy import func, or_, select, text, update
from sqlalchemy.exc import IntegrityError

from src.api.services.generation.provider_billing_policy import apply_provider_billing_policy
from src.api.services.generation.provider_failures import ProviderFailure, ProviderFailureKind
from src.api.services.image_thumbnail import make_image_thumbnails
from src.api.services.job_state_transition import GenerationOutputData
from src.api.services.media_ingest import InvalidMediaError
from src.api.services.media_ingest.profile_log import log_video_profile
from src.api.services.storage import StorageType
from src.api.services.thumbnail import extract_video_thumbnail
from src.api.services.video_jobs.errors import ProviderOutputUnavailableError
from src.api.services.video_jobs.outcome import VideoPollOutcome
from src.core.enums import JobStatus, MediaFormat, VideoPollStatus
from src.core.library_ref import LibraryAssetSource, format_asset_ref
from src.core.uid import new_id
from src.db.models import GenerationJob, GenerationMaterializationAttempt, GenerationOutput

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.generation.provider_billing_policy import ProviderBillingPolicyRegistry
    from src.api.services.job_state_transition import JobStateTransitionService
    from src.api.services.media_ingest import MediaIngestor, VideoStreamProfile
    from src.api.services.media_ingest.profile_log import VideoProfileOrigin
    from src.api.services.storage import R2StorageService
    from src.api.services.video_jobs.outcome import RemoteVideoResult
    from src.core.enums import Provider

logger = structlog.get_logger(__name__)

_RETRYABLE_CLIENT_ERRORS = frozenset({408, 425, 429})
"""4xx statuses that mean "try again later" rather than "this URL is gone"."""


def _is_permanent_status(status_code: int) -> bool:
    """Whether a result-URL HTTP status can never heal inside the poll window.

    A 404/403/410 on a provider's presigned URL is an expired or invalid signature;
    retrying only delays the same outcome until the job is refunded as a timeout.
    """
    return 400 <= status_code < 500 and status_code not in _RETRYABLE_CLIENT_ERRORS


class VideoJobNotFoundError(Exception):
    """Raised when a video job disappears while it is being finalized."""


@dataclasses.dataclass(frozen=True)
class _MaterializedVideo:
    """R2 objects ready to be attached by the claim-owning transaction."""

    outputs: list[GenerationOutputData]
    storage_keys: list[str]
    stream_profile: VideoStreamProfile
    """Probed facts of the stored original; logged once the output row is committed."""
    attempt_id: UUID | None = None
    product_id: str | None = None


class RemoteVideoMaterializer:
    """Claims, downloads, stores and attaches one provider's completed video."""

    def __init__(
        self,
        *,
        provider: Provider,
        profile_origin: VideoProfileOrigin,
        storage: R2StorageService,
        media_ingestor: MediaIngestor,
        http_client_provider: Callable[[], httpx.AsyncClient],
        retention_days: int,
        finalization_lease_seconds: int,
        max_output_bytes: int,
        billing_policy: ProviderBillingPolicyRegistry,
        transition_service_factory: Callable[[AsyncSession], JobStateTransitionService],
    ) -> None:
        """Bind the provider-specific collaborators; no settings lookups happen later.

        ``max_output_bytes`` caps the download while it streams; it must match the
        byte budget of ``media_ingestor`` (the provider-output ingest service).
        """
        self._provider = provider
        self._profile_origin = profile_origin
        self._storage = storage
        self._media_ingestor = media_ingestor
        self._http_client_provider = http_client_provider
        self._retention_days = retention_days
        self._finalization_lease_seconds = finalization_lease_seconds
        self._max_output_bytes = max_output_bytes
        self._billing_policy = billing_policy
        self._transition_service_factory = transition_service_factory

    async def finalize_completed_video(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        result: RemoteVideoResult,
        product_id: str,
        settle_failure: Callable[
            [AsyncSession, UUID, VideoPollOutcome, str], Awaitable[GenerationJob]
        ],
    ) -> GenerationJob:
        """Claim, materialize, and attach one completed remote video.

        The claim transaction is intentionally separate and committed before
        the CDN/R2 work.  A process crash leaves a short, durable lease that a
        later poller can recover; a losing poller has no R2 write or output
        row to commit.
        """
        claim_token = await self._claim_video_finalization(session, job_id)
        if claim_token is None:
            return await self._refresh_authoritative_job(session, job_id)

        materialized: _MaterializedVideo | None = None
        try:
            materialized = await self._materialize_video_result(
                user_id=(await self._refresh_authoritative_job(session, job_id)).user_id,
                job_id=job_id,
                result=result,
                product_id=product_id,
                session=session,
                claim_token=claim_token,
            )
            transition_service = self._transition_service_factory(session)
            (
                settled_job,
                did_transition,
            ) = await transition_service.transition_claimed_video_to_completed(
                job_id,
                claim_token=claim_token,
                outputs=materialized.outputs,
                product_id=product_id,
                materialization_attempt_id=materialized.attempt_id,
            )
            if did_transition:
                logger.info(
                    "video_job.job_completed",
                    provider=self._provider.value,
                    job_id=str(job_id),
                    output_count=len(materialized.outputs),
                )
                self._log_video_profile(job_id, materialized)
            else:
                # Another terminal transition won.  Never delete based on a
                # stale in-memory list: first reconcile DB references, then
                # queue only the proven-unreferenced attempt keys.
                await self._reconcile_materialization_attempt(
                    session,
                    job_id=job_id,
                    product_id=product_id,
                    materialized=materialized,
                )
        except asyncio.CancelledError:
            # Cancellation is a BaseException, so it must be handled before
            # the ordinary error path.  Shield cleanup/release so the task is
            # still discoverable even while shutdown is cancelling workers.
            if materialized is not None:
                await asyncio.shield(
                    self._reconcile_materialization_attempt(
                        session,
                        job_id=job_id,
                        product_id=product_id,
                        materialized=materialized,
                    )
                )
            else:
                await asyncio.shield(
                    self._release_video_finalization_claim(session, job_id, claim_token)
                )
            raise
        except IntegrityError:
            await session.rollback()
            if materialized is not None:
                await self._reconcile_materialization_attempt(
                    session,
                    job_id=job_id,
                    product_id=product_id,
                    materialized=materialized,
                )
            else:
                await self._release_video_finalization_claim(session, job_id, claim_token)
            return await self._refresh_authoritative_job(session, job_id)
        except InvalidMediaError:
            await session.rollback()
            if materialized is not None:
                await self._reconcile_materialization_attempt(
                    session,
                    job_id=job_id,
                    product_id=product_id,
                    materialized=materialized,
                )
                raise
            logger.exception(
                "video_job.output_invalid", provider=self._provider.value, job_id=str(job_id)
            )
            return await self._settle_undelivered(
                session,
                job_id=job_id,
                product_id=product_id,
                claim_token=claim_token,
                settle_failure=settle_failure,
            )
        except ProviderOutputUnavailableError:
            await session.rollback()
            if materialized is not None:
                # Raised only by the download, which precedes the attempt row.
                raise RuntimeError(
                    "ProviderOutputUnavailableError after materialization began"
                ) from None
            return await self._settle_undelivered(
                session,
                job_id=job_id,
                product_id=product_id,
                claim_token=claim_token,
                settle_failure=settle_failure,
            )
        except Exception:
            # ``commit()`` can raise after PostgreSQL has committed, and a
            # post-commit refresh can fail too.  Re-read in a new transaction
            # before deciding whether any R2 key is safe to remove.
            await session.rollback()
            if materialized is not None:
                await self._reconcile_materialization_attempt(
                    session,
                    job_id=job_id,
                    product_id=product_id,
                    materialized=materialized,
                )
            else:
                await self._release_video_finalization_claim(session, job_id, claim_token)
            raise
        else:
            return settled_job

    async def _settle_undelivered(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        product_id: str,
        claim_token: str,
        settle_failure: Callable[
            [AsyncSession, UUID, VideoPollOutcome, str], Awaitable[GenerationJob]
        ],
    ) -> GenerationJob:
        """Settle a completed-upstream job whose output could not be obtained.

        Used when nothing was materialized: the provider finished (and billed) the
        generation but its result is unusable or no longer served, so the job fails
        as ``OUTPUT_NOT_DELIVERED`` under the provider's undelivered-output billing
        policy instead of being retried into a refunded timeout.
        """
        # The ordinary failure transition deliberately cannot preempt a live
        # finalization claim. Relinquish our own claim before using that
        # existing settlement path.
        await self._release_video_finalization_claim(session, job_id, claim_token)
        failure = apply_provider_billing_policy(
            ProviderFailure(
                kind=ProviderFailureKind.OUTPUT_NOT_DELIVERED,
                provider=self._provider,
                sanitized_message=ProviderFailure.safe_message_for_kind(
                    ProviderFailureKind.OUTPUT_NOT_DELIVERED
                ),
                provider_request_accepted=True,
            ),
            registry=self._billing_policy,
        )
        return await settle_failure(
            session,
            job_id,
            VideoPollOutcome(
                status=VideoPollStatus.FAILED,
                error_message=failure.sanitized_message,
                failure=failure,
            ),
            product_id,
        )

    async def _claim_video_finalization(self, session: AsyncSession, job_id: UUID) -> str | None:
        """Durably acquire a recoverable PostgreSQL claim for one video job."""
        token = str(uuid4())
        result = await session.execute(
            update(GenerationJob)
            .where(
                GenerationJob.id == job_id,
                GenerationJob.status.in_([JobStatus.QUEUED.value, JobStatus.RUNNING.value]),
                or_(
                    GenerationJob.finalization_claim_token.is_(None),
                    GenerationJob.finalization_lease_expires_at <= func.now(),
                ),
            )
            .values(
                finalization_claim_token=token,
                # PostgreSQL time is authoritative for both the takeover
                # predicate and expiry.  Host-clock skew cannot create an
                # immediately expired (or unexpectedly long) lease.
                finalization_lease_expires_at=(
                    func.now()
                    + text("(:lease_seconds * interval '1 second')").bindparams(
                        lease_seconds=self._finalization_lease_seconds
                    )
                ),
            )
            .execution_options(synchronize_session=False)
        )
        if result.rowcount == 0:  # type: ignore[attr-defined]
            return None
        await session.commit()
        return token

    async def _release_video_finalization_claim(
        self, session: AsyncSession, job_id: UUID, claim_token: str
    ) -> None:
        """Release a failed materializer's claim so retries need not await TTL."""
        try:
            await session.execute(
                update(GenerationJob)
                .where(
                    GenerationJob.id == job_id,
                    GenerationJob.finalization_claim_token == claim_token,
                )
                .values(finalization_claim_token=None, finalization_lease_expires_at=None)
            )
            await session.commit()
        except Exception:
            await session.rollback()
            logger.warning(
                "video_job.finalization_claim_release_failed",
                provider=self._provider.value,
                job_id=str(job_id),
            )

    async def _refresh_authoritative_job(
        self, session: AsyncSession, job_id: UUID
    ) -> GenerationJob:
        job = await session.get(GenerationJob, job_id)
        if job is None:
            raise VideoJobNotFoundError(f"Job {job_id} not found")
        await session.refresh(job)
        return job

    def _log_video_profile(self, job_id: UUID, materialized: _MaterializedVideo) -> None:
        """Log the stored video original's stream facts (the first, non-thumbnail output)."""
        # Invariant: a failed upload of position 0 (the original) raises, so a materialized
        # result always carries the original first.
        original = materialized.outputs[0]
        log_video_profile(
            origin=self._profile_origin,
            asset_ref=format_asset_ref(LibraryAssetSource.OUTPUT, original.id),
            profile=materialized.stream_profile,
            width=original.width,
            height=original.height,
            duration_ms=original.duration_ms,
            job_id=job_id,
        )

    async def _materialize_video_result(
        self,
        *,
        user_id: UUID,
        job_id: UUID,
        result: RemoteVideoResult,
        product_id: str | None = None,
        session: AsyncSession | None = None,
        claim_token: str | None = None,
    ) -> _MaterializedVideo:
        """Download/upload artifacts after recording their planned R2 keys.

        The durable attempt is inserted and committed before the first R2
        upload.  It contains every planned key, not just successful uploads,
        because a process can die after R2 accepts ``put_raw`` but before the
        following database acknowledgement reaches this process.
        """
        video_data = await self._download_video(result.url, job_id)
        prepared = await self._media_ingestor.prepare_video(video_data)
        expires_at = datetime.now(UTC) + timedelta(days=self._retention_days)
        output_id = new_id()
        storage_key = self._storage.build_storage_key(
            user_id=user_id,
            file_id=output_id,
            storage_type=StorageType.OUTPUT,
            format=prepared.format,
            job_id=job_id,
        )
        artifacts: list[tuple[GenerationOutputData, bytes]] = [
            (
                GenerationOutputData(
                    id=output_id,
                    storage_key=storage_key,
                    content_type=prepared.content_type,
                    size_bytes=len(prepared.data),
                    format=prepared.format.value,
                    output_index=0,
                    expires_at=expires_at,
                    width=prepared.width,
                    height=prepared.height,
                    duration_ms=prepared.duration_ms,
                    hash_set=prepared.hash_set,
                ),
                prepared.data,
            )
        ]

        # Derive all thumbnails before the first upload so their keys are also
        # durable before they can exist in R2. Thumbnail generation is still
        # best-effort; a failed upload merely omits that output record.
        frame_bytes = await extract_video_thumbnail(prepared.data)
        if frame_bytes:
            for generated in await make_image_thumbnails(frame_bytes):
                thumbnail_id = new_id()
                thumbnail_key = self._storage.build_storage_key(
                    user_id=user_id,
                    file_id=thumbnail_id,
                    storage_type=StorageType.OUTPUT,
                    format=MediaFormat.WEBP,
                    job_id=job_id,
                )
                artifacts.append(
                    (
                        GenerationOutputData(
                            id=thumbnail_id,
                            storage_key=thumbnail_key,
                            content_type=MediaFormat.WEBP.content_type,
                            size_bytes=len(generated.result.data),
                            format=MediaFormat.WEBP.value,
                            output_index=0,
                            expires_at=expires_at,
                            is_thumbnail=True,
                            parent_output_id=output_id,
                            thumbnail_max_edge=generated.spec.max_edge,
                            width=generated.result.width,
                            height=generated.result.height,
                        ),
                        generated.result.data,
                    )
                )
        else:
            logger.warning(
                "video_job.thumbnail_skipped", provider=self._provider.value, job_id=str(job_id)
            )

        attempt_id: UUID | None = None
        if session is not None and claim_token is not None:
            if product_id is None:
                raise ValueError("product_id is required for durable materialization attempts")
            attempt_id = await self._create_materialization_attempt(
                session,
                job_id=job_id,
                product_id=product_id,
                claim_token=claim_token,
                planned_storage_keys=[artifact.storage_key for artifact, _ in artifacts],
            )

        uploaded_keys: list[str] = []
        persisted_outputs: list[GenerationOutputData] = []
        try:
            for position, (artifact, payload) in enumerate(artifacts):
                try:
                    await self._put_output_object(
                        artifact.storage_key,
                        payload,
                        artifact.content_type,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    if position == 0:
                        raise
                    logger.warning(
                        "video_job.thumbnail_skipped",
                        provider=self._provider.value,
                        job_id=str(job_id),
                        max_edge=artifact.thumbnail_max_edge,
                    )
                    continue
                uploaded_keys.append(artifact.storage_key)
                persisted_outputs.append(artifact)
                if attempt_id is not None and session is not None and product_id is not None:
                    await self._record_materialization_upload(
                        session,
                        attempt_id,
                        product_id=product_id,
                        uploaded_storage_keys=uploaded_keys,
                    )
            return _MaterializedVideo(
                outputs=persisted_outputs,
                storage_keys=uploaded_keys,
                attempt_id=attempt_id,
                product_id=product_id,
                stream_profile=prepared.stream_profile,
            )
        except BaseException:
            if attempt_id is not None and session is not None and product_id is not None:
                await asyncio.shield(
                    self._mark_materialization_attempt_cleanup_pending(
                        session,
                        attempt_id=attempt_id,
                        product_id=product_id,
                        error="materialization interrupted before outputs were committed",
                    )
                )
                # No output transition has started on this path, so every
                # uploaded key is provably unreferenced.  Try immediately,
                # but keep the durable attempt pending if R2 is unavailable.
                await asyncio.shield(self._cleanup_materialized_objects(uploaded_keys))
            else:
                await self._cleanup_materialized_objects(uploaded_keys)
            raise

    async def _download_video(self, url: str, job_id: UUID) -> bytes:
        """Stream the provider video, aborting as soon as it passes the byte budget.

        Raises:
            InvalidMediaError: The body is larger than ``max_output_bytes``. Raised
                before anything is buffered past the cap, and before the durable
                materialization attempt exists.
            ProviderOutputUnavailableError: The URL answered with a permanent 4xx
                (see ``_is_permanent_status``); also raised before the attempt exists.
            httpx.HTTPStatusError: Any other error status (5xx, 408, 425, 429); the
                caller retries on the next poll.
        """
        limit = self._max_output_bytes
        async with self._http_client_provider().stream("GET", url) as response:
            if response.status_code >= 400:
                if _is_permanent_status(response.status_code):
                    logger.warning(
                        "video_job.output_unavailable",
                        provider=self._provider.value,
                        job_id=str(job_id),
                        http_status=response.status_code,
                    )
                    raise ProviderOutputUnavailableError(response.status_code)
                raise httpx.HTTPStatusError(
                    f"provider result URL answered HTTP {response.status_code}",
                    request=response.request,
                    response=response,
                )
            declared = response.headers.get("content-length")
            if declared is not None and declared.isdigit() and int(declared) > limit:
                self._log_output_too_large(job_id, content_length=int(declared))
                raise InvalidMediaError("provider video exceeds the configured output byte limit")
            buffer = bytearray()
            async for chunk in response.aiter_bytes():
                buffer.extend(chunk)
                if len(buffer) > limit:
                    self._log_output_too_large(job_id, content_length=None)
                    raise InvalidMediaError(
                        "provider video exceeds the configured output byte limit"
                    )
        return bytes(buffer)

    def _log_output_too_large(self, job_id: UUID, *, content_length: int | None) -> None:
        """Log an over-budget output. Never includes the (presigned) URL."""
        logger.warning(
            "video_job.output_too_large",
            provider=self._provider.value,
            job_id=str(job_id),
            content_length=content_length,
            max_output_bytes=self._max_output_bytes,
        )

    async def _put_output_object(self, key: str, data: bytes, content_type: str) -> None:
        """Store a generated artifact through R2's public service boundary."""
        await self._storage.put_raw(key, data, content_type=content_type)

    async def _create_materialization_attempt(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        product_id: str,
        claim_token: str,
        planned_storage_keys: list[str],
    ) -> UUID:
        """Commit attempt evidence before the first external storage write."""
        attempt_id = new_id()
        session.add(
            GenerationMaterializationAttempt(
                id=attempt_id,
                job_id=job_id,
                product_id=product_id,
                claim_token=claim_token,
                state="planned",
                planned_storage_keys=planned_storage_keys,
                uploaded_storage_keys=[],
            )
        )
        await session.commit()
        return attempt_id

    async def _record_materialization_upload(
        self,
        session: AsyncSession,
        attempt_id: UUID,
        *,
        product_id: str,
        uploaded_storage_keys: list[str],
    ) -> None:
        """Persist upload progress; planned keys cover acknowledgement loss."""
        await session.execute(
            update(GenerationMaterializationAttempt)
            .where(
                GenerationMaterializationAttempt.id == attempt_id,
                GenerationMaterializationAttempt.product_id == product_id,
            )
            .values(
                state="uploading",
                uploaded_storage_keys=list(uploaded_storage_keys),
                updated_at=func.now(),
            )
        )
        await session.commit()

    async def _mark_materialization_attempt_cleanup_pending(
        self,
        session: AsyncSession,
        *,
        attempt_id: UUID,
        product_id: str,
        error: str | None = None,
    ) -> None:
        """Leave durable cleanup work for a restart-safe reconciliation pass."""
        try:
            await session.rollback()
            await session.execute(
                update(GenerationMaterializationAttempt)
                .where(
                    GenerationMaterializationAttempt.id == attempt_id,
                    GenerationMaterializationAttempt.product_id == product_id,
                )
                .values(
                    state="cleanup_pending",
                    reconciliation_error=error,
                    updated_at=func.now(),
                )
            )
            await session.commit()
        except Exception:
            await session.rollback()
            logger.exception(
                "video_job.materialization_attempt_mark_cleanup_failed",
                provider=self._provider.value,
                attempt_id=str(attempt_id),
            )

    async def _reconcile_materialization_attempt(
        self,
        session: AsyncSession,
        *,
        job_id: UUID,
        product_id: str,
        materialized: _MaterializedVideo,
    ) -> None:
        """Use DB truth before deleting objects after an uncertain transition.

        The caller may have lost a commit acknowledgement or failed while
        refreshing a committed row.  Referenced keys are preserved.  When the
        transaction outcome is not provable, the attempt remains ``ambiguous``
        for the reconciliation worker instead of deleting media optimistically.
        """
        if materialized.attempt_id is None:
            # Compatibility with direct unit-test helpers: without a durable
            # attempt we can still safely delete only after a DB reference
            # query, but cannot promise restart recovery.
            attempt_keys = materialized.storage_keys
        else:
            await session.rollback()
            attempt = await session.scalar(
                select(GenerationMaterializationAttempt).where(
                    GenerationMaterializationAttempt.id == materialized.attempt_id,
                    GenerationMaterializationAttempt.product_id == product_id,
                )
            )
            attempt_keys = list(attempt.planned_storage_keys) if attempt is not None else []

        try:
            job = await self._refresh_authoritative_job(session, job_id)
            referenced_keys = set(
                (
                    await session.scalars(
                        select(GenerationOutput.storage_key).where(
                            GenerationOutput.job_id == job_id,
                            GenerationOutput.product_id == product_id,
                        )
                    )
                ).all()
            )
        except Exception:
            if materialized.attempt_id is not None:
                await self._mark_materialization_attempt_cleanup_pending(
                    session,
                    attempt_id=materialized.attempt_id,
                    product_id=product_id,
                    error="could not establish database commit outcome",
                )
                await session.execute(
                    update(GenerationMaterializationAttempt)
                    .where(
                        GenerationMaterializationAttempt.id == materialized.attempt_id,
                        GenerationMaterializationAttempt.product_id == product_id,
                    )
                    .values(state="ambiguous", updated_at=func.now())
                )
                await session.commit()
            logger.exception(
                "video_job.materialization_reconcile_unknown",
                provider=self._provider.value,
                job_id=str(job_id),
            )
            return

        unreferenced = [key for key in attempt_keys if key not in referenced_keys]
        if job.status == JobStatus.COMPLETED.value and not unreferenced:
            if materialized.attempt_id is not None:
                await session.execute(
                    update(GenerationMaterializationAttempt)
                    .where(
                        GenerationMaterializationAttempt.id == materialized.attempt_id,
                        GenerationMaterializationAttempt.product_id == product_id,
                    )
                    .values(state="committed", completed_at=func.now(), updated_at=func.now())
                )
                await session.commit()
            return

        if materialized.attempt_id is not None:
            await self._mark_materialization_attempt_cleanup_pending(
                session,
                attempt_id=materialized.attempt_id,
                product_id=product_id,
                error="attempt has unreferenced artifact keys" if referenced_keys else None,
            )
        # These keys are proven not to be attached to an output row in the
        # authoritative read.  A failure remains durable through the attempt.
        try:
            await self._storage.delete_many(unreferenced)
        except Exception:
            logger.exception(
                "video_job.materialization_cleanup_failed",
                provider=self._provider.value,
                count=len(unreferenced),
                job_id=str(job_id),
            )
            return

        if materialized.attempt_id is not None:
            await session.execute(
                update(GenerationMaterializationAttempt)
                .where(
                    GenerationMaterializationAttempt.id == materialized.attempt_id,
                    GenerationMaterializationAttempt.product_id == product_id,
                )
                .values(state="cleaned", reconciliation_error=None, updated_at=func.now())
            )
            await session.commit()

    async def _cleanup_materialized_objects(self, storage_keys: list[str]) -> None:
        if not storage_keys:
            return
        try:
            await self._storage.delete_many(storage_keys)
        except Exception:
            # Do not hide an already-known DB/materialization failure.  The
            # retention/orphan worker remains the final safety net.
            logger.exception(
                "video_job.materialization_cleanup_failed",
                provider=self._provider.value,
                count=len(storage_keys),
            )
