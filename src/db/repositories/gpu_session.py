"""Repository for GPU session database operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import or_, select, update

from src.core.enums import TERMINAL_GPU_SESSION_STATUSES, GpuSessionStatus
from src.db.models.gpu_session import GpuSession

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession


class GpuSessionRepository:
    """Repository for GPU session CRUD operations."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        id: UUID,
        user_id: UUID,
        product_id: str,
        status: GpuSessionStatus | str,
        bundle_name: str,
        model_type: str,
        bundle_version: str | None = None,
        cf_tunnel_id: str | None = None,
        cf_dns_record_id: str | None = None,
        tunnel_hostname: str | None = None,
        vastai_instance_id: int | None = None,
        vastai_offer_id: int | None = None,
        vastai_cost_per_hour_micros: int | None = None,
        vastai_gpu_name: str | None = None,
        vastai_machine_id: int | None = None,
        callback_token_hash: str | None = None,
        account_id: UUID | None = None,
        readiness_marker_node_class: str | None = None,
        bootstrap_operation_id: UUID | None = None,
    ) -> GpuSession:
        """Create and persist a new GPU session row.

        Args:
            id: Session UUID (caller provides, e.g. via UUIDv7).
            user_id: Owner user.
            product_id: Product scope.
            status: Initial GpuSessionStatus value (enum member preferred).
            bundle_name: ai-bundles bundle name.
            model_type: ModelType slug.
            bundle_version: Pinned bundle version; None = 'current' symlink.
            cf_tunnel_id: Cloudflare tunnel ID (set after tunnel creation).
            cf_dns_record_id: Cloudflare DNS record ID.
            tunnel_hostname: Full tunnel hostname for routing.
            vastai_instance_id: Vast.ai instance ID once provisioned.
            vastai_offer_id: Vast.ai offer ID selected.
            vastai_cost_per_hour_micros: Hourly cost in microdollars.
            vastai_gpu_name: GPU model name from Vast.ai.
            vastai_machine_id: Vast.ai physical machine id of the selected offer.
            callback_token_hash: SHA-256 hex digest of the callback bearer token.
            account_id: Billing account ID for charging; may be None if not yet determined.
            bootstrap_operation_id: Bootstrap operation created with this session.

        Returns:
            Created and flushed GpuSession instance.
        """
        session_row = GpuSession(
            id=id,
            user_id=user_id,
            product_id=product_id,
            status=status,
            bundle_name=bundle_name,
            bundle_version=bundle_version,
            model_type=model_type,
            cf_tunnel_id=cf_tunnel_id,
            cf_dns_record_id=cf_dns_record_id,
            tunnel_hostname=tunnel_hostname,
            vastai_instance_id=vastai_instance_id,
            vastai_offer_id=vastai_offer_id,
            vastai_cost_per_hour_micros=vastai_cost_per_hour_micros,
            vastai_gpu_name=vastai_gpu_name,
            vastai_machine_id=vastai_machine_id,
            callback_token_hash=callback_token_hash,
            account_id=account_id,
            readiness_marker_node_class=readiness_marker_node_class,
            bootstrap_operation_id=bootstrap_operation_id,
        )
        self._session.add(session_row)
        await self._session.flush()
        return session_row

    async def get_by_id(
        self,
        session_id: UUID,
        *,
        for_update: bool = False,
    ) -> GpuSession | None:
        """Get a session by primary key.

        Args:
            session_id: Session UUID.
            for_update: When True, appends FOR UPDATE to the query.

        Returns:
            GpuSession or None.
        """
        if not for_update:
            return await self._session.get(GpuSession, session_id)

        result = await self._session.execute(
            select(GpuSession).where(GpuSession.id == session_id).with_for_update()
        )
        return result.scalar_one_or_none()

    async def get_by_id_for_user(
        self,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
    ) -> GpuSession | None:
        """Get a session scoped to a specific user and product.

        Args:
            session_id: Session UUID.
            user_id: Owner filter.
            product_id: Product filter.

        Returns:
            GpuSession or None if not found / not owned.
        """
        result = await self._session.execute(
            select(GpuSession).where(
                GpuSession.id == session_id,
                GpuSession.user_id == user_id,
                GpuSession.product_id == product_id,
            )
        )
        return result.scalar_one_or_none()

    async def get_active_for_model(
        self,
        user_id: UUID,
        product_id: str,
        model_type: str,
    ) -> GpuSession | None:
        """Get the user's active session for a model type (status='active' only).

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            model_type: ModelType slug filter.

        Returns:
            Active GpuSession or None.
        """
        result = await self._session.execute(
            select(GpuSession).where(
                GpuSession.user_id == user_id,
                GpuSession.product_id == product_id,
                GpuSession.model_type == model_type,
                GpuSession.status == GpuSessionStatus.active,
            )
        )
        return result.scalar_one_or_none()

    async def get_non_terminal_for_model(
        self,
        user_id: UUID,
        product_id: str,
        model_type: str,
    ) -> GpuSession | None:
        """Get any non-terminal session for this (user, product, model_type).

        Used before creating a new session to enforce the uniqueness constraint
        at the application layer before hitting the DB partial unique index.

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            model_type: ModelType slug filter.

        Returns:
            Non-terminal GpuSession or None.
        """
        result = await self._session.execute(
            select(GpuSession).where(
                GpuSession.user_id == user_id,
                GpuSession.product_id == product_id,
                GpuSession.model_type == model_type,
                GpuSession.status.not_in(tuple(TERMINAL_GPU_SESSION_STATUSES)),
            )
        )
        return result.scalar_one_or_none()

    async def list_by_user(
        self,
        user_id: UUID,
        product_id: str,
        *,
        include_terminal: bool = False,
    ) -> Sequence[GpuSession]:
        """List sessions for a user, optionally excluding terminal states.

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            include_terminal: When False, excludes 'stopped' and 'failed'.

        Returns:
            Sequence of GpuSession ordered by created_at DESC.
        """
        query = select(GpuSession).where(
            GpuSession.user_id == user_id,
            GpuSession.product_id == product_id,
        )
        if not include_terminal:
            query = query.where(GpuSession.status.not_in(tuple(TERMINAL_GPU_SESSION_STATUSES)))

        result = await self._session.execute(query.order_by(GpuSession.created_at.desc()))
        return result.scalars().all()

    async def list_by_status(self, *statuses: GpuSessionStatus | str) -> Sequence[GpuSession]:
        """List sessions matching any of the given statuses.

        Used by the provisioning worker to poll pending/provisioning/resuming sessions.

        Args:
            *statuses: One or more GpuSessionStatus values (enum members preferred).

        Returns:
            Sequence of matching GpuSession rows.
        """
        result = await self._session.execute(
            select(GpuSession).where(GpuSession.status.in_(statuses))
        )
        return result.scalars().all()

    async def list_all_admin(self, *, limit: int = 100) -> Sequence[GpuSession]:
        """Admin: list all sessions across all users and products.

        Args:
            limit: Maximum rows to return.

        Returns:
            Sequence of GpuSession ordered by created_at DESC.
        """
        result = await self._session.execute(
            select(GpuSession).order_by(GpuSession.created_at.desc()).limit(limit)
        )
        return result.scalars().all()

    async def increment_provision_attempt(self, session_id: UUID) -> int:
        """Atomic increment of provision_attempt; returns the new value.

        Args:
            session_id: Session to update.

        Returns:
            New provision_attempt value after increment.
        """
        result = await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(provision_attempt=GpuSession.provision_attempt + 1)
            .returning(GpuSession.provision_attempt)
        )
        await self._session.flush()
        return result.scalar_one()

    async def update_instance(
        self,
        session_id: UUID,
        *,
        vastai_instance_id: int,
        vastai_offer_id: int,
        vastai_cost_per_hour_micros: int,
        vastai_gpu_name: str,
        provisioning_started_at: datetime,
        vastai_machine_id: int | None = None,
    ) -> None:
        """Swap instance info on a session after a retry; status unchanged (stays 'pending').

        Args:
            session_id: Session to update.
            vastai_instance_id: New Vast.ai instance ID.
            vastai_offer_id: New offer ID.
            vastai_cost_per_hour_micros: New hourly cost.
            vastai_gpu_name: New GPU model name.
            provisioning_started_at: Reset timestamp (restarts the timeout window).
            vastai_machine_id: New Vast.ai physical machine id.
        """
        await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(
                vastai_instance_id=vastai_instance_id,
                vastai_offer_id=vastai_offer_id,
                vastai_cost_per_hour_micros=vastai_cost_per_hour_micros,
                vastai_gpu_name=vastai_gpu_name,
                vastai_machine_id=vastai_machine_id,
                provisioning_started_at=provisioning_started_at,
            )
        )
        await self._session.flush()

    async def add_paused_seconds(self, session_id: UUID, seconds: int) -> None:
        """Add to the cumulative total_paused_seconds counter."""
        await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(total_paused_seconds=GpuSession.total_paused_seconds + seconds)
        )
        await self._session.flush()

    async def list_pending_billing_finalization(
        self,
        *,
        grace_cutoff: datetime,
        limit: int,
    ) -> Sequence[GpuSession]:
        """List stopped sessions whose billing has not been finalized.

        Filters:
        - status = 'stopped' (terminal — the only status _finalize_billing applies to)
        - billing_finalized_at IS NULL (not yet successfully finalized)
        - stopped_at < grace_cutoff (skip in-flight in-line retries)

        Ordered oldest-first so the longest-stuck sessions reconcile first
        on each sweep. Bounded by ``limit`` to cap per-sweep work.
        """
        result = await self._session.execute(
            select(GpuSession)
            .where(
                GpuSession.status == GpuSessionStatus.stopped,
                GpuSession.billing_finalized_at.is_(None),
                # NULL stopped_at skips the grace period — include unconditionally
                # (stopped sessions should always have stopped_at, but be defensive).
                or_(GpuSession.stopped_at.is_(None), GpuSession.stopped_at < grace_cutoff),
            )
            .order_by(GpuSession.stopped_at.asc())
            .limit(limit)
        )
        return result.scalars().all()

    async def increment_billing_finalization_attempts(self, session_id: UUID) -> int:
        """Bump the attempt counter and return the new value.

        Called by the reconciler after each failed sweep to track repeat
        failures for quarantine alerting.
        """
        result = await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(billing_finalization_attempts=GpuSession.billing_finalization_attempts + 1)
            .returning(GpuSession.billing_finalization_attempts)
        )
        new_count = result.scalar_one()
        await self._session.flush()
        return new_count

    async def mark_billing_finalized(self, session_id: UUID, finalized_at: datetime) -> None:
        """Stamp ``billing_finalized_at`` to mark a session as billing-complete.

        Called by ``GpuSessionService._finalize_billing`` on a successful
        overage debit / partial refund / no-op. A NULL value indicates the
        billing finalization has not run (or failed both retries) and a
        reconciler worker should pick it up.
        """
        await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(billing_finalized_at=finalized_at)
        )
        await self._session.flush()

    async def list_orphaned_instance_candidates(
        self,
        *,
        terminal_statuses: Sequence[GpuSessionStatus],
        stopped_before: datetime,
        created_after: datetime,
        limit: int = 50,
    ) -> Sequence[GpuSession]:
        """Sessions in terminal state where we may have failed to destroy the instance.

        Filter: status IN terminal_statuses
            AND vastai_instance_id IS NOT NULL
            AND vastai_instance_destroyed_at IS NULL
            AND (stopped_at IS NULL OR stopped_at < stopped_before)  -- grace period
            AND created_at > created_after                           -- horizon cap

        ``limit`` caps per-sweep work so a backlog doesn't blow the sweep budget.
        """
        stmt = (
            select(GpuSession)
            .where(
                GpuSession.status.in_(terminal_statuses),
                GpuSession.vastai_instance_id.isnot(None),
                GpuSession.vastai_instance_destroyed_at.is_(None),
                or_(GpuSession.stopped_at.is_(None), GpuSession.stopped_at < stopped_before),
                GpuSession.created_at > created_after,
            )
            .order_by(GpuSession.created_at.asc())
            .limit(limit)
        )
        result = await self._session.execute(stmt)
        return result.scalars().all()

    async def mark_instance_destroyed(self, session_id: UUID, destroyed_at: datetime) -> None:
        """Stamp vastai_instance_destroyed_at to record confirmed instance destruction.

        Called on successful destroy_instance() — both from in-flow teardown and
        the orphan sweeper. A NULL value means the instance may still be running.
        """
        await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(vastai_instance_destroyed_at=destroyed_at)
        )
        await self._session.flush()

    async def update_credit_warning(
        self,
        session_id: UUID,
        level: str | None,
        warned_at: datetime | None,
    ) -> None:
        """Persist the current credit warning level and timestamp.

        Pass level=None to clear the warning (de-escalate on top-up).
        """
        await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(credit_warning_level=level, credit_warned_at=warned_at)
        )
        await self._session.flush()

    async def update_callback_token_hash(self, session_id: UUID, token_hash: str) -> None:
        """Replace the stored callback token hash (called on provisioning retry)."""
        await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(callback_token_hash=token_hash)
        )
        await self._session.flush()

    async def update_bootstrap_operation_id(self, session_id: UUID, operation_id: UUID) -> None:
        """Point a session at the bootstrap operation; does not synchronize in-session objects."""
        await self._session.execute(
            update(GpuSession)
            .where(GpuSession.id == session_id)
            .values(bootstrap_operation_id=operation_id)
        )
        await self._session.flush()

    async def touch_last_progress(self, session_id: UUID, at: datetime) -> None:
        """Advance the bootstrap stall-detector clock after an accepted event."""
        await self._session.execute(
            update(GpuSession).where(GpuSession.id == session_id).values(last_progress_at=at)
        )
        await self._session.flush()

    async def update_status(
        self,
        session_id: UUID,
        status: GpuSessionStatus | str,
        **extra_fields: object,
    ) -> None:
        """Update session status and any additional fields atomically.

        Args:
            session_id: Session to update.
            status: New GpuSessionStatus value (enum member preferred).
            **extra_fields: Additional model field assignments (e.g. started_at=...).
        """
        values: dict[str, Any] = {"status": status, **extra_fields}
        await self._session.execute(
            update(GpuSession).where(GpuSession.id == session_id).values(**values)
        )
        await self._session.flush()
