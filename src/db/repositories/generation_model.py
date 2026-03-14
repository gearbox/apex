"""Repository for GenerationModel database operations."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.generation_model import GenerationModel

__all__ = ["GenerationModelRepository"]

logger = structlog.get_logger(__name__)


class GenerationModelRepository:
    """Repository for per-model enable/disable state."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def list_all(self) -> Sequence[GenerationModel]:
        """Return all models ordered by provider, model_key."""
        result = await self._session.execute(
            select(GenerationModel).order_by(GenerationModel.provider, GenerationModel.model_key)
        )
        return result.scalars().all()

    async def list_enabled(self) -> Sequence[GenerationModel]:
        """Return only enabled models ordered by provider, model_key."""
        result = await self._session.execute(
            select(GenerationModel)
            .where(GenerationModel.is_enabled.is_(True))
            .order_by(GenerationModel.provider, GenerationModel.model_key)
        )
        return result.scalars().all()

    async def get_by_key(self, model_key: str) -> GenerationModel | None:
        """Fetch a single model by its model_key."""
        result = await self._session.execute(
            select(GenerationModel).where(GenerationModel.model_key == model_key)
        )
        return result.scalar_one_or_none()

    async def get_by_model_key(self, model_key: str) -> GenerationModel | None:
        """Fetch a single model by its model_key (alias for get_by_key)."""
        return await self.get_by_key(model_key)

    async def set_enabled(self, model_key: str, is_enabled: bool) -> GenerationModel | None:
        """Toggle is_enabled flag. Updates updated_at. Returns updated record or None."""
        result = await self._session.execute(
            select(GenerationModel).where(GenerationModel.model_key == model_key)
        )
        obj = result.scalar_one_or_none()
        if obj is None:
            return None
        obj.is_enabled = is_enabled
        obj.updated_at = datetime.now(UTC)
        await self._session.flush()
        logger.info(
            "generation_model.set_enabled",
            model_key=model_key,
            is_enabled=is_enabled,
        )
        return obj
