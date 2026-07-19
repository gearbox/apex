"""Database model for library asset metadata (favorite, display title)."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    String,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.uid import new_id
from src.db.models.base import Base

__all__ = ["LibraryAssetMetadata"]


class LibraryAssetMetadata(Base):
    """Per-asset library metadata: favorite flag and display title.

    Polymorphic by design (asset_type + asset_id) — a single library asset
    can back either a UserImage or a GenerationOutput row, so no FK
    relationship() is declared here; ownership and existence of the
    underlying asset are checked by the service layer before any row in
    this table is created or read. Rows are created lazily on first
    mutation (favorite/rename), not eagerly backfilled (D8).
    """

    __tablename__ = "library_asset_metadata"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=new_id,
    )
    product_id: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
    )
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    asset_type: Mapped[str] = mapped_column(String(10), nullable=False)
    asset_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)

    display_title: Mapped[str | None] = mapped_column(String(255), nullable=True)
    is_favorite: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default=text("false"),
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
        onupdate=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        UniqueConstraint(
            "product_id",
            "user_id",
            "asset_type",
            "asset_id",
            name="uq_library_asset_metadata_asset",
        ),
        CheckConstraint(
            "asset_type IN ('upload', 'output')",
            name="ck_library_asset_metadata_asset_type",
        ),
        Index(
            "ix_library_asset_metadata_favorites",
            "user_id",
            "product_id",
            postgresql_where=text("is_favorite"),
        ),
    )

    def __repr__(self) -> str:
        return (
            f"<LibraryAssetMetadata {self.id} user={self.user_id} "
            f"asset={self.asset_type}:{self.asset_id}>"
        )
