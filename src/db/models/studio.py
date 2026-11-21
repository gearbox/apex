"""Database models for the Media Project Studio (``src/studio``).

Graph::

    users ─┬─< studio_artists ─< studio_artist_memories   (CASCADE; scoped through the artist)
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
* ``studio_runs`` (planner runs, P2a) hangs off a track (``ON DELETE CASCADE``) and points at the
  versions it produced with ``ON DELETE SET NULL``. A run is terminal once: the CHECKs tie
  ``finished_at`` to the terminal statuses and a ``failed`` run to its ``error_code``.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    Date,
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
    StudioMemoryOrigin,
    StudioMemoryStatus,
    StudioReleaseKind,
    StudioRunKind,
    StudioRunStage,
    StudioRunStatus,
)
from src.core.uid import new_id
from src.db.models.base import Base
from src.studio.constraints import (
    MEMORY_IMAGE_MAX,
    MEMORY_THEME_MAX,
    MEMORY_TITLE_MAX,
)

__all__ = [
    "RUN_ONE_ACTIVE_PER_TRACK",
    "RUN_USER_IDEMPOTENCY",
    "StudioArtist",
    "StudioArtistMemory",
    "StudioDocumentVersion",
    "StudioProjectProfile",
    "StudioRun",
    "StudioTrack",
]

RUN_ONE_ACTIVE_PER_TRACK = "uq_studio_runs_one_active_per_track"
"""Partial unique index: at most one queued/running run per track. The authority for 409."""
RUN_USER_IDEMPOTENCY = "uq_studio_runs_user_idempotency"
"""``UNIQUE (user_id, idempotency_key)``: a replayed submit is the same run."""


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


class StudioArtistMemory(Base):
    """One episode of an artist's (fictional) life, used by planner runs as inspiration (P2b).

    Mutable, like the artist it belongs to. It carries **no** ``user_id`` / ``product_id`` of its
    own: ownership and product scoping are inherited through ``artist_id`` (P2-M1), so every
    repository read joins the artist, and ``users → studio_artists → memories`` cascades on a user
    hard-delete. Planner runs copy the text they use into their snapshot (P2-M4), so editing or
    deleting a memory never changes a run that has started. Lock order: artist, then memory.
    """

    __tablename__ = "studio_artist_memories"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    artist_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_artists.id", ondelete="CASCADE"),
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(MEMORY_TITLE_MAX), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    """At most ``MEMORY_BODY_MAX`` characters, enforced by the service (the column is ``text``)."""
    occurred_on: Mapped[date | None] = mapped_column(Date, nullable=True)
    """A fictional date. Used for ordering and display only."""
    themes: Mapped[list[str]] = mapped_column(
        ARRAY(String(MEMORY_THEME_MAX)), nullable=False, server_default=text("'{}'")
    )
    imagery: Mapped[list[str]] = mapped_column(
        ARRAY(String(MEMORY_IMAGE_MAX)), nullable=False, server_default=text("'{}'")
    )
    salience: Mapped[int] = mapped_column(SmallInteger, nullable=False, server_default=text("3"))
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=StudioMemoryStatus.ACTIVE.value
    )
    origin: Mapped[str] = mapped_column(
        String(20), nullable=False, server_default=StudioMemoryOrigin.USER.value
    )

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        CheckConstraint("salience BETWEEN 1 AND 5", name="ck_studio_artist_memories_salience"),
        CheckConstraint(
            _in_list("status", StudioMemoryStatus), name="ck_studio_artist_memories_status"
        ),
        CheckConstraint(
            _in_list("origin", StudioMemoryOrigin), name="ck_studio_artist_memories_origin"
        ),
        # The list: newest first, per artist.
        Index(
            "ix_studio_artist_memories_list",
            "artist_id",
            text("created_at DESC"),
            text("id DESC"),
        ),
        # Selection (P2-M2): active memories by salience, then recency (undated last), then id.
        Index(
            "ix_studio_artist_memories_selection",
            "artist_id",
            text("salience DESC"),
            text("occurred_on DESC NULLS LAST"),
            "id",
            postgresql_where=text("status = 'active'"),
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioArtistMemory {self.id} artist={self.artist_id}>"


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


_TERMINAL_RUN_STATUSES = ", ".join(f"'{m.value}'" for m in sorted(StudioRunStatus.terminal()))
_ACTIVE_RUN_STATUSES = ", ".join(f"'{m.value}'" for m in sorted(StudioRunStatus.active()))


class StudioRun(Base):
    """One planner run: an LLM job that writes a new plan or lyrics version of a track.

    The row is the whole state machine (``queued → running → succeeded | failed``,
    ``queued → cancelled``); every transition is a guarded ``UPDATE … WHERE status = …`` that
    bumps ``revision``. ``request`` / ``context_snapshot`` hold user text, so nothing outside the
    owner-scoped repository reads them and no log or ``RunOut`` carries them.
    """

    __tablename__ = "studio_runs"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    track_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("studio_tracks.id", ondelete="CASCADE"), nullable=False
    )

    kind: Mapped[str] = mapped_column(String(20), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    stage: Mapped[str | None] = mapped_column(String(20), nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("1"))

    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    request: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    inputs: Mapped[dict[str, Any] | None] = mapped_column(JSONB(none_as_null=True), nullable=True)
    """What the run works on (``RunInputs``), resolved when it was accepted; SQL ``NULL`` for
    ``plan`` (``none_as_null``: without it ``None`` is written as the JSON value ``null``).

    Ids and the language only - never document text. Read by the claim and by the delete guard
    (a version an active run takes as input cannot be deleted)."""
    context_snapshot: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    prompt_version: Mapped[str | None] = mapped_column(String(40), nullable=True)
    model_key: Mapped[str] = mapped_column(String(60), nullable=False)
    model_id: Mapped[str | None] = mapped_column(String(120), nullable=True)

    price_tokens: Mapped[int] = mapped_column(Integer, nullable=False)
    billing_account_id: Mapped[UUID | None] = mapped_column(PG_UUID(as_uuid=True), nullable=True)

    result_plan_version_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    result_lyrics_version_id: Mapped[UUID | None] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_document_versions.id", ondelete="SET NULL"),
        nullable=True,
    )
    warnings: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_detail: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB(none_as_null=True), nullable=True
    )
    """Issue paths and codes of a failure that has any; SQL ``NULL`` (not the JSON value ``null``)
    for every other failure and for a run that did not fail."""

    llm_calls: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    prompt_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default=text("0"))
    completion_tokens: Mapped[int] = mapped_column(
        Integer, nullable=False, server_default=text("0")
    )

    lease_expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        UniqueConstraint("user_id", "idempotency_key", name=RUN_USER_IDEMPOTENCY),
        CheckConstraint(_in_list("kind", StudioRunKind), name="ck_studio_runs_kind"),
        CheckConstraint(_in_list("status", StudioRunStatus), name="ck_studio_runs_status"),
        CheckConstraint(
            "stage IS NULL OR " + _in_list("stage", StudioRunStage), name="ck_studio_runs_stage"
        ),
        CheckConstraint("price_tokens >= 0", name="ck_studio_runs_price"),
        CheckConstraint("revision >= 1", name="ck_studio_runs_revision"),
        CheckConstraint(
            f"(status IN ({_TERMINAL_RUN_STATUSES})) = (finished_at IS NOT NULL)",
            name="ck_studio_runs_finished",
        ),
        CheckConstraint(
            "status <> 'failed' OR error_code IS NOT NULL", name="ck_studio_runs_failed_code"
        ),
        CheckConstraint("(kind = 'plan') = (inputs IS NULL)", name="ck_studio_runs_inputs_by_kind"),
        CheckConstraint(
            "(kind = 'plan' OR result_plan_version_id IS NULL)"
            " AND (kind <> 'plan' OR result_lyrics_version_id IS NULL)",
            name="ck_studio_runs_result_by_kind",
        ),
        Index(
            RUN_ONE_ACTIVE_PER_TRACK,
            "track_id",
            unique=True,
            postgresql_where=text(f"status IN ({_ACTIVE_RUN_STATUSES})"),
        ),
        Index(
            "ix_studio_runs_queued",
            "created_at",
            postgresql_where=text("status = 'queued'"),
        ),
        Index(
            "ix_studio_runs_running_lease",
            "lease_expires_at",
            postgresql_where=text("status = 'running'"),
        ),
        Index("ix_studio_runs_user_created", "user_id", text("created_at DESC")),
        Index(
            "ix_studio_runs_track_created",
            "track_id",
            text("created_at DESC"),
            text("id DESC"),
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioRun {self.id} track={self.track_id} {self.kind} {self.status}>"
