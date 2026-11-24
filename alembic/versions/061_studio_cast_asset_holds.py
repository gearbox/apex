"""Studio cast and library asset holds (S1-P3).

Revision ID: 061
Revises: 060
Create Date: 2026-11-24 00:00:00.000000

Five new tables and one new column (see ``src/db/models/library.py::LibraryAssetHold`` and
``src/db/models/studio_cast.py`` for the rationale):

* ``library_asset_holds`` - the library's pin mechanism, owned by the library and opaque about its
  holders (``holder_kind`` / ``holder_id``). Exactly one of ``upload_id`` / ``output_id`` is set
  (CHECK). Both asset columns are typed foreign keys with **``ON DELETE NO ACTION``**, never
  ``RESTRICT``: the database itself then refuses to delete a held asset, and a user hard-delete
  (which removes holds and assets in one cascading statement) still succeeds because ``NO ACTION``
  is checked at the end of the statement. The two partial unique indexes make "the same holder
  holds the same asset twice" unrepresentable **and** serve the ``EXISTS (hold)`` lookups by asset
  (the retention sweep, the library ``held`` flag), so no extra index on the asset columns exists.
* ``studio_cast_members`` - ``uq_studio_cast_members_id_owner`` is the target of the composite
  owner FKs below and of the artist's performer link; the name is unique per owner by
  ``lower(name)`` like artists'.
* ``studio_cast_references`` - a user-labelled reference. ``hold_id`` is ``NOT NULL UNIQUE`` (FK to
  ``library_asset_holds``, ``NO ACTION``): "every reference is pinned" is a database invariant, and
  the asset is read through the hold (no second copy of it). ``(cast_member_id, position)`` is
  ``DEFERRABLE INITIALLY DEFERRED`` so a reorder can rewrite many rows in one transaction.
* ``studio_cast_looks`` and ``studio_cast_look_references`` - a look is a named subset of a
  member's references; deleting a reference removes it from every look (the look may end up empty).
* ``studio_artists.performer_cast_member_id`` - an optional composite FK to the member
  (``fk_studio_artists_performer``, ``NO ACTION``: deleting a performer is refused, never silently
  unlinked), with a partial index for the "is this member in use" lookup.

The CHECK and FK names are spelled out here because a migration never imports ``src``.

Downgrade drops the artist column and then every cast member, reference, look and hold (the holds
of **every** holder, not only the cast's: nothing but the cast uses them yet, and the table goes
with the revision), in reverse order. The pinned assets themselves are untouched.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "061"
down_revision: str | Sequence[str] | None = "060"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TS = sa.DateTime(timezone=True)
_NOW = sa.text("CURRENT_TIMESTAMP")
_UUID = postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    """Upgrade database schema."""
    # -- library_asset_holds -------------------------------------------------------------------
    op.create_table(
        "library_asset_holds",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("upload_id", _UUID, nullable=True),
        sa.Column("output_id", _UUID, nullable=True),
        sa.Column("holder_kind", sa.String(48), nullable=False),
        sa.Column("holder_id", _UUID, nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["upload_id"],
            ["user_images.id"],
            name="fk_library_asset_holds_upload_id",
            ondelete="NO ACTION",
        ),
        sa.ForeignKeyConstraint(
            ["output_id"],
            ["generation_outputs.id"],
            name="fk_library_asset_holds_output_id",
            ondelete="NO ACTION",
        ),
        sa.CheckConstraint(
            "num_nonnulls(upload_id, output_id) = 1", name="ck_library_asset_holds_one_asset"
        ),
    )
    op.create_index("ix_library_asset_holds_product_id", "library_asset_holds", ["product_id"])
    op.create_index("ix_library_asset_holds_user_id", "library_asset_holds", ["user_id"])
    op.create_index(
        "uq_library_asset_holds_upload_holder",
        "library_asset_holds",
        ["upload_id", "holder_kind", "holder_id"],
        unique=True,
        postgresql_where=sa.text("upload_id IS NOT NULL"),
    )
    op.create_index(
        "uq_library_asset_holds_output_holder",
        "library_asset_holds",
        ["output_id", "holder_kind", "holder_id"],
        unique=True,
        postgresql_where=sa.text("output_id IS NOT NULL"),
    )
    op.create_index(
        "ix_library_asset_holds_holder", "library_asset_holds", ["holder_kind", "holder_id"]
    )

    # -- cast members --------------------------------------------------------------------------
    op.create_table(
        "studio_cast_members",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("appearance", sa.String(320), nullable=True),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("id", "user_id", "product_id", name="uq_studio_cast_members_id_owner"),
    )
    op.create_index("ix_studio_cast_members_product_id", "studio_cast_members", ["product_id"])
    op.create_index("ix_studio_cast_members_user_id", "studio_cast_members", ["user_id"])
    op.create_index(
        "uq_studio_cast_members_owner_name",
        "studio_cast_members",
        ["user_id", "product_id", sa.text("lower(name)")],
        unique=True,
    )
    op.create_index(
        "ix_studio_cast_members_owner_created",
        "studio_cast_members",
        ["user_id", "product_id", "created_at", "id"],
    )

    # -- cast references -----------------------------------------------------------------------
    op.create_table(
        "studio_cast_references",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("cast_member_id", _UUID, nullable=False),
        sa.Column("label", sa.String(40), nullable=False),
        sa.Column("position", sa.SmallInteger(), nullable=False),
        sa.Column("hold_id", _UUID, nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["cast_member_id", "user_id", "product_id"],
            [
                "studio_cast_members.id",
                "studio_cast_members.user_id",
                "studio_cast_members.product_id",
            ],
            name="fk_studio_cast_references_member_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["hold_id"],
            ["library_asset_holds.id"],
            name="fk_studio_cast_references_hold",
            ondelete="NO ACTION",
        ),
        sa.CheckConstraint("position >= 0", name="ck_studio_cast_references_position"),
        sa.UniqueConstraint("hold_id", name="uq_studio_cast_references_hold_id"),
        sa.UniqueConstraint(
            "cast_member_id",
            "position",
            name="uq_studio_cast_references_position",
            deferrable=True,
            initially="DEFERRED",
        ),
    )
    op.create_index(
        "ix_studio_cast_references_product_id", "studio_cast_references", ["product_id"]
    )
    op.create_index("ix_studio_cast_references_user_id", "studio_cast_references", ["user_id"])
    op.create_index(
        "uq_studio_cast_references_member_label",
        "studio_cast_references",
        ["cast_member_id", sa.text("lower(label)")],
        unique=True,
    )

    # -- cast looks ----------------------------------------------------------------------------
    op.create_table(
        "studio_cast_looks",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("cast_member_id", _UUID, nullable=False),
        sa.Column("name", sa.String(60), nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["cast_member_id", "user_id", "product_id"],
            [
                "studio_cast_members.id",
                "studio_cast_members.user_id",
                "studio_cast_members.product_id",
            ],
            name="fk_studio_cast_looks_member_owner",
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_studio_cast_looks_product_id", "studio_cast_looks", ["product_id"])
    op.create_index("ix_studio_cast_looks_user_id", "studio_cast_looks", ["user_id"])
    op.create_index(
        "uq_studio_cast_looks_member_name",
        "studio_cast_looks",
        ["cast_member_id", sa.text("lower(name)")],
        unique=True,
    )

    op.create_table(
        "studio_cast_look_references",
        sa.Column("look_id", _UUID, nullable=False),
        sa.Column("reference_id", _UUID, nullable=False),
        sa.PrimaryKeyConstraint("look_id", "reference_id"),
        sa.ForeignKeyConstraint(["look_id"], ["studio_cast_looks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["reference_id"], ["studio_cast_references.id"], ondelete="CASCADE"
        ),
    )
    op.create_index(
        "ix_studio_cast_look_references_reference",
        "studio_cast_look_references",
        ["reference_id"],
    )

    # -- the artist's performer link -----------------------------------------------------------
    op.add_column("studio_artists", sa.Column("performer_cast_member_id", _UUID, nullable=True))
    op.create_foreign_key(
        "fk_studio_artists_performer",
        "studio_artists",
        "studio_cast_members",
        ["performer_cast_member_id", "user_id", "product_id"],
        ["id", "user_id", "product_id"],
        ondelete="NO ACTION",
    )
    op.create_index(
        "ix_studio_artists_performer",
        "studio_artists",
        ["performer_cast_member_id"],
        postgresql_where=sa.text("performer_cast_member_id IS NOT NULL"),
    )


def downgrade() -> None:
    """Downgrade database schema (drops every cast member, reference, look and hold)."""
    # Reverse order: the artist column points at members; references point at members and holds;
    # the look links point at looks and references.
    op.drop_index("ix_studio_artists_performer", table_name="studio_artists")
    op.drop_constraint("fk_studio_artists_performer", "studio_artists", type_="foreignkey")
    op.drop_column("studio_artists", "performer_cast_member_id")
    op.drop_table("studio_cast_look_references")
    op.drop_table("studio_cast_looks")
    op.drop_table("studio_cast_references")
    op.drop_table("studio_cast_members")
    op.drop_table("library_asset_holds")
