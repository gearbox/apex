"""Base repository with shared data access patterns."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from sqlalchemy import literal, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.base import Base

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from uuid import UUID


class BaseRepository[ModelT: Base]:
    """Provides common data access patterns for all repositories.

    Subclasses must set ``_model`` to the SQLAlchemy model class.

    Assumptions:
        The model has ``id``, ``user_id``, and ``created_at`` columns.
        All three current subclasses (Job, Output, UserImage) satisfy this.
    """

    _model: type[ModelT]

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def _get_with_optional_owner(
        self,
        pk: UUID,
        *,
        user_id: UUID | None = None,
    ) -> ModelT | None:
        """Fetch a single record by PK, optionally scoped to a user.

        When ``user_id`` is ``None`` this delegates to ``session.get()``
        which benefits from SQLAlchemy's identity-map cache. When
        ``user_id`` is provided a compound ``WHERE`` is issued instead
        so that ownership is checked in a single round trip.

        Args:
            pk: Primary key value.
            user_id: Optional owner filter.

        Returns:
            Model instance or ``None``.
        """
        if user_id is None:
            return cast(ModelT | None, await self._session.get(self._model, pk))

        result = await self._session.execute(
            select(self._model).where(
                self._model.id == pk,  # type: ignore[attr-defined]
                self._model.user_id == user_id,  # type: ignore[attr-defined]
            )
        )
        return result.scalar_one_or_none()

    async def _list_by_user_cursor(
        self,
        user_id: UUID,
        *,
        limit: int,
        cursor_ts: datetime | None = None,
        cursor_id: UUID | None = None,
    ) -> Sequence[ModelT]:
        """List records for a user with keyset (cursor) pagination.

        Uses the limit+1 fetch pattern — caller checks
        ``len(result) > limit`` to determine ``has_more``.

        Results are ordered by ``(created_at DESC, id DESC)``.

        Args:
            user_id: Owner filter.
            limit: Page size (fetches limit+1 internally).
            cursor_ts: ``created_at`` of the last item on the previous page.
            cursor_id: ``id`` of the last item on the previous page.

        Returns:
            Sequence of model instances.
        """
        model = self._model
        query = select(model).where(
            model.user_id == user_id,  # type: ignore[attr-defined]
        )

        if cursor_ts is not None and cursor_id is not None:
            query = query.where(
                tuple_(
                    model.created_at,  # type: ignore[attr-defined]
                    model.id,  # type: ignore[attr-defined]
                )
                < tuple_(literal(cursor_ts), literal(cursor_id))
            )

        result = await self._session.execute(
            query.order_by(
                model.created_at.desc(),  # type: ignore[attr-defined]
                model.id.desc(),  # type: ignore[attr-defined]
            ).limit(limit + 1)
        )
        return result.scalars().all()
