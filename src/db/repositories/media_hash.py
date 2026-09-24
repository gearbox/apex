"""Repository for staging durable media-hash ledger rows."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.db.models.media_hash import MediaHash
from src.db.repositories.base import BaseRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.ext.asyncio import AsyncSession


class MediaHashRepository(BaseRepository[MediaHash]):
    """Stages ledger rows; the owning service decides when mandatory work flushes."""

    _model = MediaHash

    def __init__(self, session: AsyncSession) -> None:
        super().__init__(session)

    def add_all(self, rows: Sequence[MediaHash]) -> None:
        self._session.add_all(rows)
