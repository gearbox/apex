"""Repository for GPU session database operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.enums import GpuSessionStatus
from src.db.models.gpu_session import GpuSession

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

_TERMINAL_STATUSES = (GpuSessionStatus.stopped, GpuSessionStatus.failed)


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
        callback_token: str | None = None,
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
            callback_token: Shared secret for Phase 2 GPU → Apex callbacks.

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
            callback_token=callback_token,
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
                GpuSession.status.not_in(_TERMINAL_STATUSES),
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
            query = query.where(GpuSession.status.not_in(_TERMINAL_STATUSES))

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

    async def update_status(
        self,
        session_id: UUID,
        status: GpuSessionStatus | str,
        **extra_fields: Any,
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
