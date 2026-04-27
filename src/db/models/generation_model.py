"""GenerationModel database model — persists per-model enable/disable state."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import Boolean, DateTime, Index, String, Text, text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.uid import new_id
from src.db.models.base import Base

__all__ = ["GenerationModel"]


class GenerationModel(Base):
    """Persisted enable/disable flag for each generation model.

    model_key matches the string value of ModelType enum.
    provider is "grok" or "aisha".
    """

    __tablename__ = "generation_models"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=new_id,
    )
    model_key: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        unique=True,
    )
    provider: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )
    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default=text("''"),
    )
    is_enabled: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default=text("true"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("clock_timestamp()"),
    )

    __table_args__ = (
        Index("ix_generation_models_provider", "provider"),
        Index("ix_generation_models_is_enabled", "is_enabled"),
    )

    def __repr__(self) -> str:
        return f"<GenerationModel {self.model_key} enabled={self.is_enabled}>"
