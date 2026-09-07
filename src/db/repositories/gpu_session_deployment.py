"""Repository for GPU session deployment database operations.

Deployments hold model identity and the (user, product, model_type) uniqueness
slot since P2 — see GpuSessionDeployment's module docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import BigInteger, Exists, Text, func, select, update
from sqlalchemy import cast as sql_cast
from sqlalchemy.orm import aliased

from src.core.enums import (
    LIVE_DEPLOYMENT_STATUSES,
    TERMINAL_COMMAND_STATUSES,
    TERMINAL_GPU_SESSION_STATUSES,
    DeploymentStatus,
    GpuSessionStatus,
    OperationKind,
)
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_command import GpuSessionCommand
from src.db.models.gpu_session_deployment import GpuSessionDeployment

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.engine import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession


# These queries run once per orchestration tick across every tenant. Oldest-first
# bounded pages keep one tick's database work flat while eventually reaching every row.
_ORCHESTRATION_CANDIDATE_LIMIT = 100
# A session normally has one deployment per supported model.  This guard is
# deliberately generous, but prevents an invariant violation from turning the
# unbounded per-session cohort query into an unbounded worker round trip.
RESTART_COHORT_SANITY_LIMIT = 1_000

# CO2 advisory-lock keyspaces: command claims hash the bare session id (see
# GPU_SESSION_COMMAND_CLAIM_ADVISORY_LOCK_NAMESPACE in gpu_session_command.py).
# Deployment removal must never contend with the claim path, whose latency is
# part of the node agent's 30-second budget.
GPU_SESSION_DEPLOYMENT_REMOVAL_ADVISORY_LOCK_NAMESPACE = "deployment:"


class GpuSessionDeploymentRepository:
    """Repository for GPU session deployment CRUD and routing queries.

    Callers own transactions — no ``commit()`` here, matching every other
    repository in this codebase.
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self,
        *,
        id: UUID,
        session_id: UUID,
        user_id: UUID,
        product_id: str,
        model_type: str,
        bundle_name: str,
        status: DeploymentStatus | str,
        bundle_version: str | None = None,
        readiness_marker_node_class: str | None = None,
        pending_restart: bool = False,
        provision_operation_id: UUID | None = None,
        is_primary: bool = False,
    ) -> GpuSessionDeployment:
        """Create and persist a new deployment row.

        Args:
            id: Deployment UUID (caller provides, e.g. via UUIDv7).
            session_id: Owning GPU session.
            user_id: Owner user (denormalized from the session).
            product_id: Product scope (denormalized from the session).
            model_type: ModelType slug.
            bundle_name: ai-bundles bundle name.
            status: Initial DeploymentStatus value (enum member preferred).
            bundle_version: Pinned bundle version; None = 'current' symlink.
            readiness_marker_node_class: ComfyUI class the readiness probe requires.
            pending_restart: Forward slot for P4; always False in P2.
            provision_operation_id: The operation that (re)provisions this deployment.
            is_primary: Whether this is the deployment created with the session.

        Returns:
            Created and flushed GpuSessionDeployment instance.
        """
        deployment = GpuSessionDeployment(
            id=id,
            session_id=session_id,
            user_id=user_id,
            product_id=product_id,
            model_type=model_type,
            bundle_name=bundle_name,
            bundle_version=bundle_version,
            readiness_marker_node_class=readiness_marker_node_class,
            status=status,
            pending_restart=pending_restart,
            provision_operation_id=provision_operation_id,
            is_primary=is_primary,
        )
        self._session.add(deployment)
        await self._session.flush()
        return deployment

    async def get_live_for_model(
        self,
        user_id: UUID,
        product_id: str,
        model_type: str,
    ) -> GpuSessionDeployment | None:
        """Get the user's live deployment for a model type, if any.

        "Live" means occupying the uniqueness slot (LIVE_DEPLOYMENT_STATUSES) —
        used before creating a new session to enforce uniqueness at the
        application layer before hitting the DB partial unique index.

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            model_type: ModelType slug filter.

        Returns:
            Live GpuSessionDeployment or None.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment).where(
                GpuSessionDeployment.user_id == user_id,
                GpuSessionDeployment.product_id == product_id,
                GpuSessionDeployment.model_type == model_type,
                GpuSessionDeployment.status.in_(tuple(LIVE_DEPLOYMENT_STATUSES)),
            )
        )
        return result.scalar_one_or_none()

    async def get_routable(
        self,
        user_id: UUID,
        product_id: str,
        model_type: str,
    ) -> tuple[GpuSession, GpuSessionDeployment] | None:
        """Get the session+deployment pair routable for generation, if any.

        One joined statement requiring GpuSession.status == active AND
        GpuSessionDeployment.status == active — not two sequential reads, which
        would open a window where a session that flips to 'stopping' between
        them still gets routed a job onto a dying node. Also requires
        routing_suspended IS FALSE (S5): a restart cohort becoming ready
        suspends routing for every active deployment on the session before its
        drain check even runs, since a restart takes the whole node down, not
        only the deployment(s) being restarted — without this a generation
        could resolve, reserve credit, and call queue_prompt just as ComfyUI
        goes down for the restart.

        Args:
            user_id: Owner filter.
            product_id: Product filter.
            model_type: ModelType slug filter.

        Returns:
            (GpuSession, GpuSessionDeployment) tuple or None.
        """
        result = await self._session.execute(
            select(GpuSession, GpuSessionDeployment)
            .join(GpuSessionDeployment, GpuSessionDeployment.session_id == GpuSession.id)
            .where(
                GpuSessionDeployment.user_id == user_id,
                GpuSessionDeployment.product_id == product_id,
                GpuSessionDeployment.model_type == model_type,
                GpuSessionDeployment.status == DeploymentStatus.active,
                GpuSessionDeployment.routing_suspended.is_(False),
                GpuSession.status == GpuSessionStatus.active,
            )
        )
        row = result.first()
        return None if row is None else (row[0], row[1])

    async def has_routing_suspension_for_model(
        self,
        user_id: UUID,
        product_id: str,
        model_type: str,
    ) -> bool:
        """Whether an otherwise-active deployment is temporarily restart-suspended.

        This is intentionally narrower than "not routable": paused, stale, and
        failed sessions are not restart-in-progress states. Generation uses it
        only to give the caller an actionable wait-and-retry message after the
        normal routable lookup returns no row.
        """
        result = await self._session.execute(
            select(
                select(GpuSessionDeployment.id)
                .join(GpuSession, GpuSessionDeployment.session_id == GpuSession.id)
                .where(
                    GpuSessionDeployment.user_id == user_id,
                    GpuSessionDeployment.product_id == product_id,
                    GpuSessionDeployment.model_type == model_type,
                    GpuSessionDeployment.status == DeploymentStatus.active,
                    GpuSessionDeployment.routing_suspended.is_(True),
                    GpuSession.status == GpuSessionStatus.active,
                )
                .exists()
            )
        )
        return bool(result.scalar())

    async def list_for_session(self, session_id: UUID) -> Sequence[GpuSessionDeployment]:
        """List all deployments for one session.

        Args:
            session_id: Session to list deployments for.

        Returns:
            Sequence of GpuSessionDeployment ordered by created_at ASC.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(GpuSessionDeployment.session_id == session_id)
            .order_by(GpuSessionDeployment.created_at.asc())
        )
        return result.scalars().all()

    async def list_for_sessions(
        self, session_ids: Iterable[UUID]
    ) -> dict[UUID, list[GpuSessionDeployment]]:
        """Fetch deployments for many sessions in one query, keyed by session id.

        Used by the sessions list endpoint to avoid an N+1 — same pattern as
        GpuSessionOperationRepository.get_many.

        Args:
            session_ids: Session ids to fetch deployments for.

        Returns:
            Mapping of session_id to its deployments (ordered by created_at ASC).
        """
        ids = tuple(session_ids)
        if not ids:
            return {}
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(GpuSessionDeployment.session_id.in_(ids))
            .order_by(GpuSessionDeployment.created_at.asc())
        )
        by_session: dict[UUID, list[GpuSessionDeployment]] = {}
        for deployment in result.scalars().all():
            by_session.setdefault(deployment.session_id, []).append(deployment)
        return by_session

    async def update_provision_operation_id(self, session_id: UUID, operation_id: UUID) -> None:
        """Repoint a session's primary deployment at a new provisioning operation.

        Used by GpuProvisioningWorker._retry_or_fail: a retry re-provisions the
        same model onto a new node without forking the deployment row — it
        stays 'deploying' throughout and only this pointer rotates, so the UI
        follows the live attempt. P2 has exactly one deployment per session,
        but filtering to ``is_primary`` is load-bearing: P4 sibling deployments
        must not be repointed at the primary model's bootstrap operation. Mirrors
        GpuSessionRepository.update_bootstrap_operation_id.
        """
        await self._session.execute(
            update(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.is_primary.is_(True),
            )
            .values(provision_operation_id=operation_id)
        )
        await self._session.flush()

    async def mark_status(
        self,
        session_id: UUID,
        *,
        from_statuses: Iterable[DeploymentStatus | str],
        to_status: DeploymentStatus | str,
        at: datetime,
    ) -> int:
        """Cascade primitive: guarded UPDATE of every deployment for a session.

        Only rows currently in ``from_statuses`` are moved to ``to_status``.
        Stamps ``activated_at`` when ``to_status`` is 'active', or
        ``removed_at`` when it is 'removed'/'failed' — mirrors the session-level
        timestamp conventions in GpuSessionRepository.update_status.

        Called from the two lifecycle chokepoints (GpuSessionService._set_status
        and GpuProvisioningWorker._transition) per D15 — never from inside
        GpuSessionRepository.update_status itself, which stays dumb data access.

        Args:
            session_id: Session whose deployments to update.
            from_statuses: Only rows in one of these statuses are moved.
            to_status: New DeploymentStatus value.
            at: Timestamp to stamp on the relevant column.

        Returns:
            Number of deployment rows updated.
        """
        values: dict[str, Any] = {"status": to_status}
        if to_status == DeploymentStatus.active:
            values["activated_at"] = at
        elif to_status in (DeploymentStatus.removed, DeploymentStatus.failed):
            values["removed_at"] = at
            # A stopped/failed session must not leave a deployment eligible for
            # a late restart outcome.  resolve_restart_outcome also guards on
            # status='deploying', but clear the dead workflow state here too.
            values["pending_restart"] = False
            values["pending_restart_since"] = None
            values["restart_operation_id"] = None
            # S5: a deployment suspended for an in-progress restart must not
            # carry that suspension into a terminal state forever. The normal
            # restart path and its orphan reconciler separately release active
            # non-members such as the primary deployment.
            values["routing_suspended"] = False

        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.session_id == session_id,
                    GpuSessionDeployment.status.in_(tuple(from_statuses)),
                )
                .values(**values)
            ),
        )
        await self._session.flush()
        return result.rowcount

    async def get_live_for_session_and_model(
        self, session_id: UUID, model_type: str
    ) -> GpuSessionDeployment | None:
        """Get the live deployment for one session+model_type pair, if any.

        Used by the P4 remove endpoint to resolve the target deployment —
        session-scoped (unlike get_live_for_model, which is user-scoped and
        used for the attach-time uniqueness pre-check).
        """
        result = await self._session.execute(
            select(GpuSessionDeployment).where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.model_type == model_type,
                GpuSessionDeployment.status.in_(tuple(LIVE_DEPLOYMENT_STATUSES)),
            )
        )
        return result.scalar_one_or_none()

    async def list_live_for_session(self, session_id: UUID) -> Sequence[GpuSessionDeployment]:
        """List every live deployment for one session (P4: last-deployment guard, D11 retain_bundles)."""
        result = await self._session.execute(
            select(GpuSessionDeployment).where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.status.in_(tuple(LIVE_DEPLOYMENT_STATUSES)),
            )
        )
        return result.scalars().all()

    async def set_provision_pointer(
        self, deployment_id: UUID, *, operation_id: UUID, batch_id: str
    ) -> None:
        """P4 attach: point a freshly-created deployment at its provision batch.

        Unlike update_provision_operation_id (scoped to is_primary, CO6), this
        targets one deployment by id — the correct pointer for a P4 sibling.
        """
        await self._session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id == deployment_id)
            .values(provision_operation_id=operation_id, batch_id=batch_id)
        )
        await self._session.flush()

    async def acquire_removal_lock(self, session_id: UUID) -> None:
        """Serialize P4 removal *and* restart-suspend transitions for one session
        within the current transaction (S5: a restart and a removal on the same
        session genuinely should not interleave, since both reason about the
        live set — reused rather than adding a third advisory-lock namespace).

        This deliberately uses a deployment-namespaced hash, not the bare
        session-id hash used by the P3 command-claim path.  See the module
        constant for the two CO2 keyspaces.
        """
        await self._session.execute(
            select(
                func.pg_advisory_xact_lock(
                    func.hashtextextended(
                        sql_cast(
                            f"{GPU_SESSION_DEPLOYMENT_REMOVAL_ADVISORY_LOCK_NAMESPACE}{session_id}",
                            Text,
                        ),
                        sql_cast(0, BigInteger),
                    )
                )
            )
        )

    async def suspend_routing_for_session(self, session_id: UUID) -> Sequence[GpuSessionDeployment]:
        """S5: close routing for every 'active' deployment on a session before
        the restart drain check runs, so the in-flight count read right after
        this is monotonically non-increasing — the property that makes D35's
        timeout an actual bound instead of a hope. Caller must hold
        ``acquire_removal_lock`` in the same transaction first. Idempotent: a
        cohort that is still draining on a later tick re-suspends a no-op set.
        Returns only rows whose flag changed, so callers can publish an exact
        deployment SSE update after their transaction commits.
        """
        result = await self._session.execute(
            update(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.status == DeploymentStatus.active,
                GpuSessionDeployment.routing_suspended.is_(False),
            )
            .values(routing_suspended=True)
            .returning(GpuSessionDeployment)
            .execution_options(populate_existing=True)
        )
        await self._session.flush()
        return result.scalars().all()

    async def clear_routing_suspended_for_session(
        self, session_id: UUID
    ) -> Sequence[GpuSessionDeployment]:
        """S5/S6: release a whole-session routing suspension once its restart
        cycle resolves and no claimed sibling can still restart the node.
        Session-scoped rather than limited to the cycle's own deployment ids:
        suspension was applied to every active deployment on the session (S5),
        including ones that were already active and are not part of the cycle
        being resolved (e.g. the primary), so releasing it must cover the same
        set.
        """
        result = await self._session.execute(
            update(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.routing_suspended.is_(True),
            )
            .values(routing_suspended=False)
            .returning(GpuSessionDeployment)
            .execution_options(populate_existing=True)
        )
        await self._session.flush()
        return result.scalars().all()

    async def fail_active_siblings_with_missing_readiness_markers(
        self,
        session_id: UUID,
        *,
        markers: Collection[str],
        at: datetime,
    ) -> Sequence[GpuSessionDeployment]:
        """Fail active non-primary deployments whose resume marker is absent.

        A paused session keeps deployments active (D17), so resume must not put
        a sibling back into the routing set merely because the primary's marker
        came back. The status predicate keeps a concurrent terminal transition
        or removal authoritative; returned rows are exactly the failures that
        should be published after the caller commits.
        """
        if not markers:
            return ()
        result = await self._session.execute(
            update(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.is_primary.is_(False),
                GpuSessionDeployment.status == DeploymentStatus.active,
                GpuSessionDeployment.readiness_marker_node_class.in_(tuple(markers)),
            )
            .values(
                status=DeploymentStatus.failed,
                removed_at=at,
                routing_suspended=False,
            )
            .returning(GpuSessionDeployment)
            .execution_options(populate_existing=True)
        )
        await self._session.flush()
        return result.scalars().all()

    async def mark_removing(
        self, deployment_id: UUID, *, require_another_active: bool = False
    ) -> bool:
        """Guardedly move an active deployment to ``removing``.

        When ``require_another_active`` is true, the last-deployment rule is
        evaluated in the same UPDATE as the state transition.  ``removing``
        deliberately does not satisfy that correlated EXISTS: a serialized
        second removal must not use the first in-progress removal as permission
        to leave the session with no active deployment.
        """
        conditions = [
            GpuSessionDeployment.id == deployment_id,
            GpuSessionDeployment.status == DeploymentStatus.active,
        ]
        if require_another_active:
            other = aliased(GpuSessionDeployment)
            another_active = (
                select(other.id)
                .where(
                    other.session_id == GpuSessionDeployment.session_id,
                    other.id != GpuSessionDeployment.id,
                    other.status == DeploymentStatus.active,
                )
                .correlate(GpuSessionDeployment)
                .exists()
            )
            conditions.append(another_active)
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(*conditions)
                .values(status=DeploymentStatus.removing)
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    async def list_deploying_awaiting_provision_result(self) -> Sequence[GpuSessionDeployment]:
        """P4 orchestration worker step 1 candidates: siblings whose provision command
        may have finished. is_primary is excluded — the primary's 'deploying' -> 'active'
        transition is owned entirely by GpuProvisioningWorker._transition (session
        bootstrap/resume), never by a P3 command.

        Requires a terminal bundle_provision command to exist (N2). D31 only cancels
        commands on a terminal *session* transition, so a session paused mid-attach
        leaves its provision command queued with no agent to claim it — without this
        predicate that row would sit at the head of this bounded page forever,
        starving every newer deployment behind it.
        """
        terminal_provision_exists = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.deployment_id == GpuSessionDeployment.id,
                GpuSessionCommand.kind == OperationKind.bundle_provision,
                GpuSessionCommand.status.in_(tuple(TERMINAL_COMMAND_STATUSES)),
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.status == DeploymentStatus.deploying,
                GpuSessionDeployment.pending_restart.is_(False),
                GpuSessionDeployment.is_primary.is_(False),
                terminal_provision_exists,
            )
            .order_by(GpuSessionDeployment.created_at.asc())
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def resolve_provision_outcome(
        self, deployment_id: UUID, *, succeeded: bool, at: datetime
    ) -> bool:
        """P4 orchestration worker step 1: apply one deployment's provision-command outcome.

        Success (D33): stays 'deploying', flips pending_restart so it's picked up by the
        batch-restart step. Failure: 'failed' + removed_at, same as any other terminal write.
        """
        values: dict[str, Any] = (
            {"pending_restart": True, "pending_restart_since": at}
            if succeeded
            else {"status": DeploymentStatus.failed, "removed_at": at}
        )
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id == deployment_id,
                    GpuSessionDeployment.status == DeploymentStatus.deploying,
                    GpuSessionDeployment.pending_restart.is_(False),
                )
                .values(**values)
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    async def list_pending_restart_awaiting_operation(self) -> Sequence[GpuSessionDeployment]:
        """P4 orchestration worker step 2 candidates: provisioned, no restart enqueued yet.

        Excludes a row whose batch still has a non-terminal command (N2) — the same
        condition the worker used to evaluate in Python, per-batch, after fetching
        every row on the page. A batch stuck mid-provision-retry (or otherwise
        incomplete) can no longer occupy a slot on this bounded page while
        contributing nothing, which is what let a large backlog starve newer,
        genuinely ready batches.

        Excludes batch_id IS NULL (S3): that row can never match the correlated
        EXISTS above (SQL NULL = NULL is unknown), so it would otherwise pin this
        page forever. Reachable only as an invariant violation — see
        list_pending_restart_missing_batch_id, which reaps it instead.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.pending_restart.is_(True),
                GpuSessionDeployment.restart_operation_id.is_(None),
                GpuSessionDeployment.batch_id.is_not(None),
                ~self._batch_has_non_terminal_command(),
            )
            .order_by(GpuSessionDeployment.pending_restart_since.asc().nulls_first())
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def list_pending_restart_awaiting_operation_for_session(
        self, session_id: UUID
    ) -> Sequence[GpuSessionDeployment]:
        """Return the complete restart-ready cohort for a session selected by the bounded scan.

        The global scan is deliberately capped, but a restart must cover every ready
        deployment on a selected session. This narrow follow-up query prevents a
        large global backlog from splitting one session's restart across ticks, and
        carries the same batch-readiness predicate as the global scan (N2), including
        the batch_id IS NOT NULL exclusion (S3), so every row it returns is already
        known-ready — the caller no longer needs to re-derive readiness in Python.

        Fetch one row beyond the generous sanity ceiling.  The worker rejects an
        oversized cohort rather than truncating it: activating a partial cohort
        would leave some readiness markers unverified.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.pending_restart.is_(True),
                GpuSessionDeployment.restart_operation_id.is_(None),
                GpuSessionDeployment.batch_id.is_not(None),
                ~self._batch_has_non_terminal_command(),
            )
            .order_by(GpuSessionDeployment.pending_restart_since.asc().nulls_first())
            .limit(RESTART_COHORT_SANITY_LIMIT + 1)
        )
        return result.scalars().all()

    async def list_pending_restart_missing_batch_id(self) -> Sequence[GpuSessionDeployment]:
        """S3: a pending-restart row with no batch_id can never match the
        correlated EXISTS in list_pending_restart_awaiting_operation (SQL
        NULL = NULL is unknown), so without this exclusion it would sit at the
        head of that bounded page forever — the N2 starvation shape, reborn for
        an invariant-violating row. Should be unreachable in practice: both
        set_provision_pointer and recover_provision_pointer always write
        batch_id alongside the provision pointer that leads here.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.pending_restart.is_(True),
                GpuSessionDeployment.restart_operation_id.is_(None),
                GpuSessionDeployment.batch_id.is_(None),
            )
            .order_by(GpuSessionDeployment.created_at.asc())
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def fail_pending_restart_missing_batch_id(
        self, deployment_id: UUID, *, at: datetime
    ) -> bool:
        """S3: guardedly fail one invariant-violating pending-restart row so it
        is reported once instead of skipped on every tick forever."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id == deployment_id,
                    GpuSessionDeployment.pending_restart.is_(True),
                    GpuSessionDeployment.restart_operation_id.is_(None),
                    GpuSessionDeployment.batch_id.is_(None),
                )
                .values(
                    status=DeploymentStatus.failed,
                    removed_at=at,
                    pending_restart=False,
                    pending_restart_since=None,
                )
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    @staticmethod
    def _batch_has_non_terminal_command() -> Exists:
        """Correlated EXISTS: this row's batch still has a queued/claimed command.

        A NULL batch_id never matches the correlation (SQL NULL = NULL is unknown),
        so a row with no batch_id — an invariant violation the worker logs and
        skips — is left for the caller to handle rather than silently filtered here.
        """
        return (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.batch_id == GpuSessionDeployment.batch_id,
                GpuSessionCommand.status.notin_(tuple(TERMINAL_COMMAND_STATUSES)),
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )

    async def set_restart_pointer(
        self, deployment_ids: Sequence[UUID], *, operation_id: UUID
    ) -> int:
        """P4 orchestration worker step 2: stamp the one restart operation onto every
        pending-restart deployment in a batch. Guarded on restart_operation_id IS NULL so a
        crash-then-duplicate-enqueue orphans the second command instead of corrupting state."""
        if not deployment_ids:
            return 0
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id.in_(tuple(deployment_ids)),
                    GpuSessionDeployment.pending_restart.is_(True),
                    GpuSessionDeployment.restart_operation_id.is_(None),
                )
                .values(restart_operation_id=operation_id)
            ),
        )
        await self._session.flush()
        return result.rowcount

    async def list_pending_restart_awaiting_outcome(self) -> Sequence[GpuSessionDeployment]:
        """P4 orchestration worker step 3 candidates: restart enqueued, outcome not yet applied."""
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.status == DeploymentStatus.deploying,
                GpuSessionDeployment.pending_restart.is_(True),
                GpuSessionDeployment.restart_operation_id.is_not(None),
            )
            .order_by(GpuSessionDeployment.pending_restart_since.asc().nulls_first())
        )
        return result.scalars().all()

    async def resolve_restart_outcome(
        self, deployment_ids: Sequence[UUID], *, succeeded: bool, at: datetime
    ) -> int:
        """P4 orchestration worker step 3: apply the restart outcome to every deployment
        in the batch (D33/D34). Success -> active/routable; failure -> failed, no retry (D34)."""
        if not deployment_ids:
            return 0
        values: dict[str, Any] = (
            {
                "status": DeploymentStatus.active,
                "pending_restart": False,
                "activated_at": at,
                "routing_suspended": False,
            }
            if succeeded
            else {
                "status": DeploymentStatus.failed,
                "pending_restart": False,
                "removed_at": at,
                "routing_suspended": False,
            }
        )
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id.in_(tuple(deployment_ids)),
                    GpuSessionDeployment.status == DeploymentStatus.deploying,
                    GpuSessionDeployment.pending_restart.is_(True),
                )
                .values(**values)
            ),
        )
        await self._session.flush()
        return result.rowcount

    async def list_orphaned_routing_suspension_session_ids(self) -> Sequence[UUID]:
        """Find sessions whose restart suspension no longer has an owner.

        A suspension is owned while a deployment is pending restart or a restart
        command is queued/claimed.  Step 4 normally releases it; this query is the
        recovery backstop for a member that reaches a terminal state another way.
        """
        pending = aliased(GpuSessionDeployment)
        pending_restart_exists = (
            select(pending.id)
            .where(
                pending.session_id == GpuSessionDeployment.session_id,
                pending.pending_restart.is_(True),
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        non_terminal_restart_exists = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.session_id == GpuSessionDeployment.session_id,
                GpuSessionCommand.kind == OperationKind.comfyui_restart,
                GpuSessionCommand.status.notin_(tuple(TERMINAL_COMMAND_STATUSES)),
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        result = await self._session.execute(
            select(GpuSessionDeployment.session_id)
            .where(
                GpuSessionDeployment.routing_suspended.is_(True),
                ~pending_restart_exists,
                ~non_terminal_restart_exists,
            )
            .distinct()
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def clear_orphaned_routing_suspension(
        self, session_id: UUID
    ) -> Sequence[GpuSessionDeployment]:
        """Release an orphaned suspension if it still has no restart owner.

        The predicates are repeated on the UPDATE rather than trusting the prior
        candidate read, so a concurrent restart enqueue cannot be reopened by the
        reconciler between detection and repair.
        """
        pending = aliased(GpuSessionDeployment)
        pending_restart_exists = (
            select(pending.id)
            .where(
                pending.session_id == GpuSessionDeployment.session_id,
                pending.pending_restart.is_(True),
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        non_terminal_restart_exists = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.session_id == GpuSessionDeployment.session_id,
                GpuSessionCommand.kind == OperationKind.comfyui_restart,
                GpuSessionCommand.status.notin_(tuple(TERMINAL_COMMAND_STATUSES)),
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        result = await self._session.execute(
            update(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.session_id == session_id,
                GpuSessionDeployment.routing_suspended.is_(True),
                ~pending_restart_exists,
                ~non_terminal_restart_exists,
            )
            .values(routing_suspended=False)
            .returning(GpuSessionDeployment)
            .execution_options(populate_existing=True)
        )
        await self._session.flush()
        return result.scalars().all()

    async def list_removing(self) -> Sequence[GpuSessionDeployment]:
        """P4 orchestration worker step 4 candidates: removal in flight with a
        resolvable outcome.

        Requires a terminal bundle_removal command to exist (N2), for the same
        starvation reason as list_deploying_awaiting_provision_result: a removal
        command stranded queued by a mid-removal session pause must not occupy a
        slot on this bounded page forever.
        """
        terminal_removal_exists = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.deployment_id == GpuSessionDeployment.id,
                GpuSessionCommand.kind == OperationKind.bundle_removal,
                GpuSessionCommand.status.in_(tuple(TERMINAL_COMMAND_STATUSES)),
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.status == DeploymentStatus.removing,
                terminal_removal_exists,
            )
            .order_by(GpuSessionDeployment.created_at.asc())
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def list_orphaned_removals_without_command(
        self, *, created_before: datetime
    ) -> Sequence[GpuSessionDeployment]:
        """Find old ``removing`` rows that never received a removal command.

        New removals write their state transition and command in one transaction,
        so ``created_at`` is only a grace-period backstop for rows stranded by
        older code or an unexpected partial failure.  There is no normal
        committed removing-without-command window to race here.
        """
        removal_command_exists = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.deployment_id == GpuSessionDeployment.id,
                GpuSessionCommand.kind == OperationKind.bundle_removal,
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.status == DeploymentStatus.removing,
                GpuSessionDeployment.created_at < created_before,
                ~removal_command_exists,
            )
            .order_by(GpuSessionDeployment.created_at.asc())
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def restore_orphaned_removal_without_command(
        self, deployment_id: UUID, *, created_before: datetime
    ) -> bool:
        """Guardedly return a stranded removal to ``active``.

        Repeating the no-command and grace predicates in the UPDATE prevents a
        concurrent legitimate enqueue from being undone by the reaper.
        """
        removal_command_exists = (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.deployment_id == GpuSessionDeployment.id,
                GpuSessionCommand.kind == OperationKind.bundle_removal,
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id == deployment_id,
                    GpuSessionDeployment.status == DeploymentStatus.removing,
                    GpuSessionDeployment.created_at < created_before,
                    ~removal_command_exists,
                )
                .values(status=DeploymentStatus.active)
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    @staticmethod
    def _provision_command_exists() -> Exists:
        """Correlated EXISTS: a bundle_provision command was persisted for this deployment.

        Used by both the pointer-recovery candidate query and the orphan reaper (N3)
        so the reaper's predicate states, in SQL, exactly the condition its name
        implies — "no command" — rather than relying only on recovery running first
        in the same tick.
        """
        return (
            select(GpuSessionCommand.id)
            .where(
                GpuSessionCommand.deployment_id == GpuSessionDeployment.id,
                GpuSessionCommand.kind == OperationKind.bundle_provision,
            )
            .correlate(GpuSessionDeployment)
            .exists()
        )

    async def list_deployments_missing_provision_pointer(self) -> Sequence[GpuSessionDeployment]:
        """N3 pointer-recovery candidates: deploying, no pointer, might have a real command.

        attach's create() commits in its own transaction before enqueue_batch's
        commit and set_provision_pointer's separate transaction — a process death
        between the two leaves a deployment with a real, queued provision command
        but a null provision_operation_id. Unlike the orphan reaper below, this scan
        has no grace period: recovery is a no-op guarded UPDATE when no command
        exists yet (the normal millisecond-sized window), so running it every tick
        is safe and lets a real pointer loss heal on the very next tick rather than
        waiting out the reaper's grace window.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.status == DeploymentStatus.deploying,
                GpuSessionDeployment.is_primary.is_(False),
                GpuSessionDeployment.provision_operation_id.is_(None),
            )
            .order_by(GpuSessionDeployment.created_at.asc())
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def recover_provision_pointer(
        self, deployment_id: UUID, *, operation_id: UUID, batch_id: str | None
    ) -> bool:
        """N3: repair a deployment's provision pointer from its real command row.

        Guarded on provision_operation_id IS NULL so a concurrent legitimate write
        (the normal attach path finishing its own set_provision_pointer) always wins
        over this repair.
        """
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id == deployment_id,
                    GpuSessionDeployment.provision_operation_id.is_(None),
                )
                .values(provision_operation_id=operation_id, batch_id=batch_id)
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    async def list_orphaned_deployments_without_provision_command(
        self, *, created_before: datetime
    ) -> Sequence[GpuSessionDeployment]:
        """Find non-primary deployments committed before command enqueue failed.

        The grace period is owned by the caller. It leaves the normal, millisecond-sized
        create -> enqueue -> provision-pointer window alone while repairing a committed
        deployment that can otherwise permanently occupy a uniqueness slot.

        NOT EXISTS a bundle_provision command for this deployment (N3): pointer
        recovery runs earlier in the same tick, but this predicate should state what
        it means on its own — a deployment with a real, queued command must never be
        reaped just because its pointer write was lost.
        """
        result = await self._session.execute(
            select(GpuSessionDeployment)
            .where(
                GpuSessionDeployment.status == DeploymentStatus.deploying,
                GpuSessionDeployment.is_primary.is_(False),
                GpuSessionDeployment.provision_operation_id.is_(None),
                GpuSessionDeployment.created_at < created_before,
                ~self._provision_command_exists(),
            )
            .order_by(GpuSessionDeployment.created_at.asc())
            .limit(_ORCHESTRATION_CANDIDATE_LIMIT)
        )
        return result.scalars().all()

    async def fail_orphaned_deployment_without_provision_command(
        self, deployment_id: UUID, *, created_before: datetime, at: datetime
    ) -> bool:
        """Guardedly fail one deployment still stranded before command enqueue."""
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id == deployment_id,
                    GpuSessionDeployment.status == DeploymentStatus.deploying,
                    GpuSessionDeployment.is_primary.is_(False),
                    GpuSessionDeployment.provision_operation_id.is_(None),
                    GpuSessionDeployment.created_at < created_before,
                    ~self._provision_command_exists(),
                )
                .values(status=DeploymentStatus.failed, removed_at=at)
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    async def resolve_removal_outcome(
        self, deployment_id: UUID, *, succeeded: bool, at: datetime
    ) -> bool:
        """P4 orchestration worker step 4: apply one removal-command outcome.

        Success -> 'removed' (frees the slot). Failure -> back to 'active' (invariant #13);
        the bundle is still resident, so the deployment is still usable.
        """
        values: dict[str, Any] = (
            {"status": DeploymentStatus.removed, "removed_at": at}
            if succeeded
            else {"status": DeploymentStatus.active}
        )
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.id == deployment_id,
                    GpuSessionDeployment.status == DeploymentStatus.removing,
                )
                .values(**values)
            ),
        )
        await self._session.flush()
        return result.rowcount == 1

    async def mark_primary_active(self, session_id: UUID, *, at: datetime) -> int:
        """D15/D33 active-cascade, scoped to only the primary deployment.

        Used by GpuProvisioningWorker._transition instead of the general mark_status
        cascade: unlike a primary's bootstrap/resume, a P4 sibling's 'deploying' ->
        'active' transition is driven exclusively by its own restart-command outcome
        (see resolve_restart_outcome). A session-level active transition (session
        bootstrap, or resume racing a concurrent attach) must never short-circuit that
        and mark a still-provisioning sibling routable before its restart even ran.
        """
        result = cast(
            "CursorResult[Any]",
            await self._session.execute(
                update(GpuSessionDeployment)
                .where(
                    GpuSessionDeployment.session_id == session_id,
                    GpuSessionDeployment.status == DeploymentStatus.deploying,
                    GpuSessionDeployment.is_primary.is_(True),
                )
                .values(status=DeploymentStatus.active, activated_at=at)
            ),
        )
        await self._session.flush()
        return result.rowcount

    async def mark_orphaned(self, *, at: datetime) -> Sequence[tuple[UUID, UUID]]:
        """D16 self-heal: mark 'removed' any live deployment whose session is terminal.

        A deployment can only become orphaned if a third transition site forgets
        the D15 cascade — this guard makes that failure loud (the caller logs
        ``gpu_session.deployment.orphaned`` at ERROR for each pair) and
        self-healing instead of a silently leaked uniqueness slot.

        Args:
            at: Timestamp to stamp as removed_at on the orphaned rows.

        Returns:
            (deployment_id, session_id) pairs for every row just marked removed.
        """
        candidates = await self._session.execute(
            select(GpuSessionDeployment.id, GpuSessionDeployment.session_id)
            .join(GpuSession, GpuSession.id == GpuSessionDeployment.session_id)
            .where(
                GpuSessionDeployment.status.in_(tuple(LIVE_DEPLOYMENT_STATUSES)),
                GpuSession.status.in_(tuple(TERMINAL_GPU_SESSION_STATUSES)),
            )
        )
        pairs = [(row[0], row[1]) for row in candidates.all()]
        if not pairs:
            return []

        deployment_ids = [deployment_id for deployment_id, _ in pairs]
        await self._session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.id.in_(deployment_ids))
            .values(status=DeploymentStatus.removed, removed_at=at)
        )
        await self._session.flush()
        return pairs
