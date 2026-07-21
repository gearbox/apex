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
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.uid import new_id
from src.db.models.base import Base

__all__ = ["LibraryAssetMetadata", "LibraryAssetTag", "LibraryProject", "LibraryTag"]


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
    project_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("library_projects.id", ondelete="SET NULL"),
        nullable=True,
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
        Index("ix_library_asset_metadata_project", "project_id"),
    )

    def __repr__(self) -> str:
        return (
            f"<LibraryAssetMetadata {self.id} user={self.user_id} "
            f"asset={self.asset_type}:{self.asset_id}>"
        )


class LibraryProject(Base):
    """User-created grouping for library assets.

    P1: one project per asset (nullable FK on ``library_asset_metadata``) —
    many-to-many membership is deferred. No ORM relationship is declared
    to the metadata table; project membership is resolved via explicit,
    batched repository queries (see LibraryProjectRepository), matching
    the rest of the library read-model's no-lazy-load convention.
    """

    __tablename__ = "library_projects"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=new_id,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

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
        Index(
            "uq_library_projects_owner_name",
            "product_id",
            "user_id",
            text("lower(name)"),
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<LibraryProject {self.id} user={self.user_id} name={self.name!r}>"


class LibraryTag(Base):
    """User-created tag, many-to-many against library assets via LibraryAssetTag.

    Unlike LibraryProject (one project per asset), a single asset can carry
    multiple tags — see library_asset_tags. No ORM relationship() is declared
    to the join table; membership is resolved via explicit, batched
    repository queries (LibraryTagRepository), matching the rest of the
    library read-model's no-lazy-load convention.
    """

    __tablename__ = "library_tags"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=new_id,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    name: Mapped[str] = mapped_column(String(50), nullable=False)

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
        Index(
            "uq_library_tags_owner_name",
            "product_id",
            "user_id",
            text("lower(name)"),
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<LibraryTag {self.id} user={self.user_id} name={self.name!r}>"


class LibraryAssetTag(Base):
    """Join row: one asset tagged with one tag. Many-to-many (T1).

    Polymorphic by design (asset_type + asset_id), same rationale as
    LibraryAssetMetadata — no FK relationship() to the content tables. T2:
    user_id/product_id are denormalized here (not resolved via a join to
    library_tags) so every scoped query can filter directly on this table.
    """

    __tablename__ = "library_asset_tags"

    tag_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("library_tags.id", ondelete="CASCADE"),
        primary_key=True,
    )
    asset_type: Mapped[str] = mapped_column(String(10), primary_key=True)
    asset_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)

    product_id: Mapped[str] = mapped_column(String(32), nullable=False)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        CheckConstraint(
            "asset_type IN ('upload', 'output')",
            name="ck_library_asset_tags_asset_type",
        ),
        Index("ix_library_asset_tags_asset", "asset_type", "asset_id"),
        Index("ix_library_asset_tags_owner_tag", "user_id", "product_id", "tag_id"),
    )

    def __repr__(self) -> str:
        return f"<LibraryAssetTag tag={self.tag_id} asset={self.asset_type}:{self.asset_id}>"
