"""Database models for the Media Project Studio (``src/studio``).

Graph::

    users ─┬─< studio_artists
           ├─< library_projects ─1:1─ studio_project_profiles >── artist (NO ACTION)
           │                                   │ NO ACTION
           │                                   └─< studio_tracks ─< studio_document_versions
           │                                            │ composite FKs (current_plan / current_lyrics)
           │                                            └────────────────────────────────┘
           └─< (every studio table also carries user_id / product_id)

Design notes (see ``docs/contracts/studio-contract.md``):

* Versions are **immutable full snapshots** in a per-(track, kind) tree. The table has no
  ``updated_at``; the only mutation is deletion.
* Same-track integrity lives in the database: every cross-row pointer is a *composite* FK that
  includes ``track_id`` (``parent``, ``based_on_plan``, ``current_*``), so another track's version
  is unrepresentable. With MATCH SIMPLE a NULL pointer skips the check, which is intended.
* ``tracks -> profile`` and ``profile -> artist`` are NO ACTION (end-of-statement check), not
  RESTRICT: RESTRICT fires immediately and would make the user hard-delete cascade order-dependent.
* Version numbers come from per-track counters and are never reused (no ``max()+1``).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column

from src.core.enums import (
    StudioDocumentKind,
    StudioDocumentOrigin,
    StudioReleaseKind,
)
from src.core.uid import new_id
from src.db.models.base import Base

__all__ = [
    "StudioArtist",
    "StudioDocumentVersion",
    "StudioProjectProfile",
    "StudioTrack",
]


def _in_list(column: str, members: type[StrEnum]) -> str:
    """Render ``column IN ('a', 'b')`` from a StrEnum's wire values."""
    rendered = ", ".join(f"'{member.value}'" for member in members)
    return f"{column} IN ({rendered})"


def _created_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
    )


def _updated_at() -> Mapped[datetime]:
    return mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=text("CURRENT_TIMESTAMP"),
        onupdate=text("CURRENT_TIMESTAMP"),
    )


class StudioArtist(Base):
    """A reusable artist persona. Mutable; versions of plans do not snapshot it (P2 snapshots per run)."""

    __tablename__ = "studio_artists"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    name: Mapped[str] = mapped_column(String(100), nullable=False)
    persona: Mapped[str | None] = mapped_column(Text, nullable=True)
    voice_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    default_language: Mapped[str | None] = mapped_column(String(35), nullable=True)
    aesthetic_references: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, server_default=text("'{}'")
    )
    """Planner-only context. Never exported."""
    style_tags: Mapped[list[str]] = mapped_column(
        ARRAY(String(60)), nullable=False, server_default=text("'{}'")
    )
    """Exportable: merged into the Suno Style field after the plan's own tags."""
    global_avoid: Mapped[list[str]] = mapped_column(
        ARRAY(String(60)), nullable=False, server_default=text("'{}'")
    )
    """Exportable: merged into the Suno Exclude field after the plan's own avoid list."""

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        Index(
            "uq_studio_artists_owner_name",
            "product_id",
            "user_id",
            text("lower(name)"),
            unique=True,
        ),
        Index("ix_studio_artists_owner_created", "user_id", "product_id", "created_at", "id"),
    )

    def __repr__(self) -> str:
        return f"<StudioArtist {self.id} user={self.user_id}>"


class StudioProjectProfile(Base):
    """Studio extension of a ``LibraryProject`` (1:1, the PK *is* the library project id)."""

    __tablename__ = "studio_project_profiles"

    library_project_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("library_projects.id", ondelete="CASCADE"),
        primary_key=True,
    )
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )
    artist_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_artists.id"),  # NO ACTION: the 409 artist_in_use backstop
        nullable=True,
        index=True,
    )
    release_kind: Mapped[str] = mapped_column(String(10), nullable=False)
    concept: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        CheckConstraint(
            _in_list("release_kind", StudioReleaseKind),
            name="ck_studio_project_profiles_release_kind",
        ),
        Index(
            "ix_studio_project_profiles_owner_created",
            "user_id",
            "product_id",
            "created_at",
            "library_project_id",
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioProjectProfile {self.library_project_id} user={self.user_id}>"


class StudioTrack(Base):
    """One track of a studio project, with its current (plan, lyrics) pair."""

    __tablename__ = "studio_tracks"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Indexed: the users ON DELETE CASCADE finds this table's rows by user_id.
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    library_project_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_project_profiles.library_project_id"),  # NO ACTION backstop
        nullable=False,
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)

    next_plan_version_number: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )
    next_lyrics_version_number: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("1")
    )

    current_plan_version_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True
    )
    current_lyrics_version_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True
    )

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        UniqueConstraint(
            "library_project_id",
            "position",
            name="uq_studio_tracks_project_position",
            deferrable=True,
            initially="DEFERRED",
        ),
        CheckConstraint("position >= 0", name="ck_studio_tracks_position"),
        CheckConstraint(
            "current_lyrics_version_id IS NULL OR current_plan_version_id IS NOT NULL",
            name="ck_studio_tracks_lyrics_need_plan",
        ),
        # Composite so the pointed-to version must belong to *this* track.
        ForeignKeyConstraint(
            ["current_plan_version_id", "id"],
            ["studio_document_versions.id", "studio_document_versions.track_id"],
            name="fk_studio_tracks_current_plan_version",
            use_alter=True,
        ),
        ForeignKeyConstraint(
            ["current_lyrics_version_id", "id"],
            ["studio_document_versions.id", "studio_document_versions.track_id"],
            name="fk_studio_tracks_current_lyrics_version",
            use_alter=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioTrack {self.id} project={self.library_project_id} pos={self.position}>"


class StudioDocumentVersion(Base):
    """An immutable snapshot of a track's plan or lyrics document."""

    __tablename__ = "studio_document_versions"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Indexed: the users ON DELETE CASCADE finds this table's rows by user_id.
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    track_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_tracks.id", ondelete="CASCADE"),
        nullable=False,
    )
    kind: Mapped[str] = mapped_column(String(10), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, nullable=False)
    parent_version_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)
    based_on_plan_version_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True), nullable=True
    )
    schema_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    origin: Mapped[str] = mapped_column(String(10), nullable=False)
    content: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)

    created_at: Mapped[datetime] = _created_at()

    __table_args__ = (
        UniqueConstraint(
            "track_id", "kind", "version_number", name="uq_studio_doc_versions_number"
        ),
        # Targets of the composite FKs (this table's own and studio_tracks').
        UniqueConstraint("id", "track_id", name="uq_studio_doc_versions_id_track"),
        UniqueConstraint("id", "track_id", "kind", name="uq_studio_doc_versions_id_track_kind"),
        # Natural idempotency: an identical sibling is the same version. NULLS NOT DISTINCT makes
        # root versions (parent NULL) collide too — without it replay of a root create breaks.
        UniqueConstraint(
            "track_id",
            "kind",
            "parent_version_id",
            "content_hash",
            name="uq_studio_doc_versions_sibling_content",
            postgresql_nulls_not_distinct=True,
        ),
        ForeignKeyConstraint(
            ["parent_version_id", "track_id", "kind"],
            [
                "studio_document_versions.id",
                "studio_document_versions.track_id",
                "studio_document_versions.kind",
            ],
            name="fk_studio_doc_versions_parent",
        ),
        ForeignKeyConstraint(
            ["based_on_plan_version_id", "track_id"],
            ["studio_document_versions.id", "studio_document_versions.track_id"],
            name="fk_studio_doc_versions_based_on_plan",
        ),
        CheckConstraint(_in_list("kind", StudioDocumentKind), name="ck_studio_doc_versions_kind"),
        CheckConstraint(
            _in_list("origin", StudioDocumentOrigin), name="ck_studio_doc_versions_origin"
        ),
        CheckConstraint("version_number >= 1", name="ck_studio_doc_versions_number"),
        CheckConstraint("schema_version >= 1", name="ck_studio_doc_versions_schema_version"),
        CheckConstraint(
            "(kind = 'lyrics') = (based_on_plan_version_id IS NOT NULL)",
            name="ck_studio_doc_versions_based_on_iff_lyrics",
        ),
        Index("ix_studio_doc_versions_based_on", "based_on_plan_version_id"),
        Index(
            "ix_studio_doc_versions_list",
            "track_id",
            "kind",
            text("created_at DESC"),
            text("id DESC"),
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioDocumentVersion {self.id} track={self.track_id} {self.kind}#{self.version_number}>"
