"""Repository for GPU session deployment database operations.

Deployments hold model identity and the (user, product, model_type) uniqueness
slot since P2 — see GpuSessionDeployment's module docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from sqlalchemy import select, update

from src.core.enums import (
    LIVE_DEPLOYMENT_STATUSES,
    TERMINAL_GPU_SESSION_STATUSES,
    DeploymentStatus,
    GpuSessionStatus,
)
from src.db.models.gpu_session import GpuSession
from src.db.models.gpu_session_deployment import GpuSessionDeployment

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from datetime import datetime
    from uuid import UUID

    from sqlalchemy.engine import CursorResult
    from sqlalchemy.ext.asyncio import AsyncSession


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
        them still gets routed a job onto a dying node.

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
                GpuSession.status == GpuSessionStatus.active,
            )
        )
        row = result.first()
        return None if row is None else (row[0], row[1])

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
        """Repoint a session's deployment(s) at a new provisioning operation.

        Used by GpuProvisioningWorker._retry_or_fail: a retry re-provisions the
        same model onto a new node without forking the deployment row — it
        stays 'deploying' throughout and only this pointer rotates, so the UI
        follows the live attempt. P2 has exactly one deployment per session, so
        this affects only the primary; mirrors
        GpuSessionRepository.update_bootstrap_operation_id.
        """
        await self._session.execute(
            update(GpuSessionDeployment)
            .where(GpuSessionDeployment.session_id == session_id)
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
