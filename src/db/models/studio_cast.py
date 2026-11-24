"""Database models of the studio cast (S1-P3, migration 061).

Graph::

    users ─< studio_cast_members ─┬─< studio_cast_references >── library_asset_holds ──> asset
                                  │            │ (hold_id, NOT NULL UNIQUE)
                                  │            └──────────────┐
                                  └─< studio_cast_looks ─< studio_cast_look_references
                                                                (a look is a subset of references)
    studio_artists.performer_cast_member_id ──> studio_cast_members      (NO ACTION)

Every table carries its own ``user_id`` / ``product_id``. A reference and a look are tied to their
cast member by a **composite owner foreign key** ``(cast_member_id, user_id, product_id)`` (the
target is ``uq_studio_cast_members_id_owner``), so a row whose owner differs from its member's is
unrepresentable; every repository query filters the row's own scope as well.

Design notes (see ``agent_prompts/studio-s1-p3-decisions.md``):

* **One hold per reference.** The asset a reference points at lives only on its hold row
  (``library_asset_holds``, library-owned): the reference stores no ``asset_ref`` of its own, so
  there is one source of truth. ``hold_id`` is ``NOT NULL UNIQUE``: "every reference is pinned" is
  a database invariant. ``ON DELETE NO ACTION``, like the hold's own asset keys, so the user
  hard-delete cascade (which removes references, holds and assets in one statement) succeeds.
* Reference ``position`` is dense (``0..n-1``); the unique ``(cast_member_id, position)`` is
  ``DEFERRABLE INITIALLY DEFERRED`` so a reorder can rewrite many rows in one transaction.
* Deleting a reference removes it from every look (the link rows cascade); a look may end up empty
  and stays (S4 rejects an empty look when it builds a shot).
* An artist's ``performer_cast_member_id`` is ``NO ACTION``: deleting a performer is refused
  (``409 cast_member_in_use``); silently unlinking would change an artist's look unannounced.

Lock order: artist → cast member → reference / look → (assets, inside the library's hold service).
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
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
from src.db.models.studio import _created_at, _updated_at
from src.studio.constraints import (
    CAST_APPEARANCE_MAX,
    CAST_LABEL_MAX,
    CAST_LOOK_NAME_MAX,
    CAST_NAME_MAX,
)

__all__ = [
    "CAST_REFERENCE_HOLDER_KIND",
    "LOOK_NAME_UNIQUE",
    "MEMBER_NAME_UNIQUE",
    "REFERENCE_HOLD_FK",
    "REFERENCE_LABEL_UNIQUE",
    "StudioCastLook",
    "StudioCastLookReference",
    "StudioCastMember",
    "StudioCastReference",
]

CAST_REFERENCE_HOLDER_KIND = "studio.cast_reference"
"""``holder_kind`` of the library hold behind every cast reference (``holder_id`` = the reference)."""
MEMBER_NAME_UNIQUE = "uq_studio_cast_members_owner_name"
REFERENCE_LABEL_UNIQUE = "uq_studio_cast_references_member_label"
LOOK_NAME_UNIQUE = "uq_studio_cast_looks_member_name"
REFERENCE_HOLD_FK = "fk_studio_cast_references_hold"
"""The reference → hold foreign key (``NO ACTION``): a hold cannot be deleted while a reference uses
it, which is what keeps the reference and its hold paired."""


class StudioCastMember(Base):
    """A person, character or performer the user keeps references for (S1-P3)."""

    __tablename__ = "studio_cast_members"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    # Indexed: the users ON DELETE CASCADE finds this table's rows by user_id.
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(CAST_NAME_MAX), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    """At most ``CAST_DESCRIPTION_MAX`` characters, enforced by the service (the column is text)."""
    appearance: Mapped[str | None] = mapped_column(String(CAST_APPEARANCE_MAX), nullable=True)
    """A short descriptor, used verbatim as a prompt fragment from S4 on."""

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        # The target of the composite owner FKs below and of the artist's performer link.
        UniqueConstraint("id", "user_id", "product_id", name="uq_studio_cast_members_id_owner"),
        Index(
            MEMBER_NAME_UNIQUE,
            "user_id",
            "product_id",
            text("lower(name)"),
            unique=True,
        ),
        Index("ix_studio_cast_members_owner_created", "user_id", "product_id", "created_at", "id"),
    )

    def __repr__(self) -> str:
        return f"<StudioCastMember {self.id} user={self.user_id}>"


class StudioCastReference(Base):
    """A user-labelled reference image of a cast member; pins its asset through ``hold_id``."""

    __tablename__ = "studio_cast_references"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cast_member_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    """The FK is composite (see ``__table_args__``): it also pins ``user_id`` / ``product_id``."""
    label: Mapped[str] = mapped_column(String(CAST_LABEL_MAX), nullable=False)
    position: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    """Dense ``0..n-1`` within the member; the order shown to the user and fed to S4."""
    hold_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("library_asset_holds.id", name=REFERENCE_HOLD_FK, ondelete="NO ACTION"),
        nullable=False,
        unique=True,
    )
    """The library hold that pins this reference's asset; the asset is read through it."""

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["cast_member_id", "user_id", "product_id"],
            [
                "studio_cast_members.id",
                "studio_cast_members.user_id",
                "studio_cast_members.product_id",
            ],
            name="fk_studio_cast_references_member_owner",
            ondelete="CASCADE",
        ),
        CheckConstraint("position >= 0", name="ck_studio_cast_references_position"),
        # Deferred: a reorder (and the re-compaction after a delete) rewrites many rows at once.
        UniqueConstraint(
            "cast_member_id",
            "position",
            name="uq_studio_cast_references_position",
            deferrable=True,
            initially="DEFERRED",
        ),
        Index(
            REFERENCE_LABEL_UNIQUE,
            "cast_member_id",
            text("lower(label)"),
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioCastReference {self.id} member={self.cast_member_id}>"


class StudioCastLook(Base):
    """A named subset of a cast member's references (an outfit, an era, a character state)."""

    __tablename__ = "studio_cast_looks"

    id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True, default=new_id)
    product_id: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    user_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    cast_member_id: Mapped[UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    name: Mapped[str] = mapped_column(String(CAST_LOOK_NAME_MAX), nullable=False)

    created_at: Mapped[datetime] = _created_at()
    updated_at: Mapped[datetime] = _updated_at()

    __table_args__ = (
        ForeignKeyConstraint(
            ["cast_member_id", "user_id", "product_id"],
            [
                "studio_cast_members.id",
                "studio_cast_members.user_id",
                "studio_cast_members.product_id",
            ],
            name="fk_studio_cast_looks_member_owner",
            ondelete="CASCADE",
        ),
        Index(
            LOOK_NAME_UNIQUE,
            "cast_member_id",
            text("lower(name)"),
            unique=True,
        ),
    )

    def __repr__(self) -> str:
        return f"<StudioCastLook {self.id} member={self.cast_member_id}>"


class StudioCastLookReference(Base):
    """Join row: one reference belongs to one look. Both sides cascade."""

    __tablename__ = "studio_cast_look_references"

    look_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_cast_looks.id", ondelete="CASCADE"),
        primary_key=True,
    )
    reference_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("studio_cast_references.id", ondelete="CASCADE"),
        primary_key=True,
    )

    __table_args__ = (
        # The look side is the primary key's leading column; this serves the reference's cascade.
        Index("ix_studio_cast_look_references_reference", "reference_id"),
    )

    def __repr__(self) -> str:
        return f"<StudioCastLookReference look={self.look_id} reference={self.reference_id}>"
