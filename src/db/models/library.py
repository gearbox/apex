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
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.uid import new_id
from src.db.models.base import Base

__all__ = [
    "HOLD_ASSET_FKS",
    "HOLD_OUTPUT_FK",
    "HOLD_UNIQUES",
    "HOLD_UPLOAD_FK",
    "LibraryAssetHold",
    "LibraryAssetMetadata",
    "LibraryAssetTag",
    "LibraryAudioProfile",
    "LibraryProject",
    "LibraryTag",
]

HOLD_UPLOAD_FK = "fk_library_asset_holds_upload_id"
HOLD_OUTPUT_FK = "fk_library_asset_holds_output_id"
HOLD_ASSET_FKS = frozenset({HOLD_UPLOAD_FK, HOLD_OUTPUT_FK})
"""The two asset foreign keys of ``library_asset_holds`` (upload / output).

A foreign-key violation naming one of these means "a held asset was deleted" (on a delete) or "the
asset vanished" (on a hold insert): every path maps it to a typed refusal, never a 500."""
HOLD_UNIQUES = frozenset(
    {"uq_library_asset_holds_upload_holder", "uq_library_asset_holds_output_holder"}
)
"""The two partial unique indexes: a violation means one holder holds one asset twice."""


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


class LibraryAssetHold(Base):
    """A pin: some bounded context holds a library asset, so nothing may delete or sweep it.

    The library knows a holder only as an opaque ``(holder_kind, holder_id)``: ``holder_kind`` is a
    namespaced string (``"studio.cast_reference"``) and the library never imports the context
    behind it (D12). The two asset columns are typed foreign keys on purpose: **the database
    refuses to delete a held asset** (``ON DELETE NO ACTION``, checked at the end of the
    statement), so no delete path has to remember a check, and a hold insert takes ``KEY SHARE``
    on the asset row, which closes the race with a concurrent delete.

    ``NO ACTION`` and not ``RESTRICT``: a user hard-delete removes the holds and the assets in one
    cascading statement, and only ``NO ACTION`` lets that statement succeed.

    Exactly one of ``upload_id`` / ``output_id`` is set. The two partial unique indexes make "the
    same holder holds the same asset twice" unrepresentable and also serve the ``EXISTS (hold)``
    lookups by asset (the retention sweep, the library ``held`` flag).
    """

    __tablename__ = "library_asset_holds"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=new_id,
    )
    # Indexed: the users ON DELETE CASCADE finds this table's rows by user_id.
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    upload_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "user_images.id",
            name=HOLD_UPLOAD_FK,
            ondelete="NO ACTION",
        ),
        nullable=True,
    )
    output_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey(
            "generation_outputs.id",
            name=HOLD_OUTPUT_FK,
            ondelete="NO ACTION",
        ),
        nullable=True,
    )
    holder_kind: Mapped[str] = mapped_column(String(48), nullable=False)
    holder_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(upload_id, output_id) = 1",
            name="ck_library_asset_holds_one_asset",
        ),
        Index(
            "uq_library_asset_holds_upload_holder",
            "upload_id",
            "holder_kind",
            "holder_id",
            unique=True,
            postgresql_where=text("upload_id IS NOT NULL"),
        ),
        Index(
            "uq_library_asset_holds_output_holder",
            "output_id",
            "holder_kind",
            "holder_id",
            unique=True,
            postgresql_where=text("output_id IS NOT NULL"),
        ),
        Index("ix_library_asset_holds_holder", "holder_kind", "holder_id"),
    )

    def __repr__(self) -> str:
        return f"<LibraryAssetHold {self.id} holder={self.holder_kind}:{self.holder_id}>"


class LibraryAudioProfile(Base):
    """Probed facts and waveform peaks of one stored audio asset (S2a, migration 062).

    Library-owned (D12) and 1:1 with its asset: exactly one of ``upload_id`` / ``output_id`` is set
    (CHECK), each a typed foreign key with ``ON DELETE CASCADE`` (the profile has no life of its
    own: the retention sweep, a delete and a user hard-delete remove it with the asset). The
    ``output_id`` arm is unused until generated audio exists (S2b). ``duration_ms`` is **not** here:
    it is the asset row's own column, like a video's.

    ``peaks`` is ``PEAKS_BYTES`` (2,048) bytes: ``int8`` interleaved ``[min0, max0, min1, max1, ...]``
    over 1,024 buckets of the mono decode; ``peaks_version`` lets a finer zoom level (S3) coexist.
    """

    __tablename__ = "library_audio_profiles"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    upload_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("user_images.id", ondelete="CASCADE"), nullable=True
    )
    output_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("generation_outputs.id", ondelete="CASCADE"),
        nullable=True,
    )
    codec: Mapped[str] = mapped_column(String(16), nullable=False)
    sample_rate_hz: Mapped[int] = mapped_column(Integer, nullable=False)
    channels: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    bit_rate: Mapped[int | None] = mapped_column(Integer, nullable=True)
    peaks: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    peaks_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )

    __table_args__ = (
        CheckConstraint(
            "num_nonnulls(upload_id, output_id) = 1",
            name="ck_library_audio_profiles_one_asset",
        ),
        CheckConstraint("octet_length(peaks) = 2048", name="ck_library_audio_profiles_peaks_size"),
        CheckConstraint("channels BETWEEN 1 AND 8", name="ck_library_audio_profiles_channels"),
        Index(
            "uq_library_audio_profiles_upload",
            "upload_id",
            unique=True,
            postgresql_where=text("upload_id IS NOT NULL"),
        ),
        Index(
            "uq_library_audio_profiles_output",
            "output_id",
            unique=True,
            postgresql_where=text("output_id IS NOT NULL"),
        ),
    )

    def __repr__(self) -> str:
        return f"<LibraryAudioProfile {self.id} codec={self.codec}>"
