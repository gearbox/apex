"""Studio artist memories: the ``studio_artist_memories`` table.

Revision ID: 059
Revises: 058
Create Date: 2026-11-20 00:00:00.000000

One table and one constraint on an existing one (see
``src/db/models/studio.py::StudioArtistMemory`` for the rationale):

* ``uq_studio_artists_id_owner`` on ``studio_artists (id, user_id, product_id)``: the target of the
  composite FK below. ``id`` alone is already unique; a composite FK needs a unique key over exactly
  the referenced columns. It is created **before** the table and dropped **after** it.
* ``studio_artist_memories`` - episodes of an artist's fictional life that planner runs use as
  inspiration. It carries its own ``user_id`` / ``product_id`` like every other studio table, and
  the composite FK ``(artist_id, user_id, product_id) -> studio_artists (id, user_id,
  product_id)`` (``ON DELETE CASCADE``) makes a memory whose owner differs from its artist's owner
  unrepresentable. A user hard-delete reaches a memory by two cascade paths (``users`` and
  ``users -> studio_artists``); both delete the same row.
* CHECKs for ``salience`` (1-5) and for ``status`` / ``origin`` (generated from the enums in the
  model; spelled out here because a migration never imports ``src``). ``origin`` is declared in
  full now (``user``, ``chat``) so the artist chat (P2c) needs no migration.
* ``ix_studio_artist_memories_list`` serves the cursor list in **diary order** (D18):
  ``(artist_id, status, occurred_on DESC NULLS LAST, created_at DESC, id DESC)``;
  ``ix_studio_artist_memories_selection`` is a partial index over active rows in the selection
  order (``salience DESC, occurred_on DESC NULLS LAST, id``).
* ``ix_studio_artist_memories_product_id`` / ``ix_studio_artist_memories_user_id`` are the plain
  indexes ``index=True`` generates (the users ``ON DELETE CASCADE`` finds rows by ``user_id``).

``themes`` / ``imagery`` are varchar arrays, which PostgreSQL truncates silently on the bind cast:
the service's length check (after NFC) is the only guard.

Downgrade drops the table (and every memory in it), then the unique constraint.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "059"
down_revision: str | Sequence[str] | None = "058"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TS = sa.DateTime(timezone=True)
_NOW = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    """Upgrade database schema."""
    # The FK target: must exist before the table that references it.
    op.create_unique_constraint(
        "uq_studio_artists_id_owner", "studio_artists", ["id", "user_id", "product_id"]
    )
    op.create_table(
        "studio_artist_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artist_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("title", sa.String(120), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("occurred_on", sa.Date(), nullable=True),
        sa.Column(
            "themes",
            postgresql.ARRAY(sa.String(60)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "imagery",
            postgresql.ARRAY(sa.String(120)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("salience", sa.SmallInteger(), server_default=sa.text("3"), nullable=False),
        sa.Column("status", sa.String(20), server_default="active", nullable=False),
        sa.Column("origin", sa.String(20), server_default="user", nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # The memory's owner and product ARE its artist's: a mismatch is unrepresentable.
        sa.ForeignKeyConstraint(
            ["artist_id", "user_id", "product_id"],
            ["studio_artists.id", "studio_artists.user_id", "studio_artists.product_id"],
            name="fk_studio_artist_memories_artist_owner",
            ondelete="CASCADE",
        ),
        sa.CheckConstraint("salience BETWEEN 1 AND 5", name="ck_studio_artist_memories_salience"),
        sa.CheckConstraint(
            "status IN ('active', 'archived')", name="ck_studio_artist_memories_status"
        ),
        sa.CheckConstraint("origin IN ('user', 'chat')", name="ck_studio_artist_memories_origin"),
    )
    op.create_index(
        "ix_studio_artist_memories_product_id", "studio_artist_memories", ["product_id"]
    )
    op.create_index("ix_studio_artist_memories_user_id", "studio_artist_memories", ["user_id"])
    op.create_index(
        "ix_studio_artist_memories_list",
        "studio_artist_memories",
        [
            "artist_id",
            "status",
            sa.text("occurred_on DESC NULLS LAST"),
            sa.text("created_at DESC"),
            sa.text("id DESC"),
        ],
    )
    op.create_index(
        "ix_studio_artist_memories_selection",
        "studio_artist_memories",
        [
            "artist_id",
            sa.text("salience DESC"),
            sa.text("occurred_on DESC NULLS LAST"),
            "id",
        ],
        postgresql_where=sa.text("status = 'active'"),
    )


def downgrade() -> None:
    """Downgrade database schema (drops every artist memory and its data)."""
    op.drop_index("ix_studio_artist_memories_selection", table_name="studio_artist_memories")
    op.drop_index("ix_studio_artist_memories_list", table_name="studio_artist_memories")
    op.drop_index("ix_studio_artist_memories_user_id", table_name="studio_artist_memories")
    op.drop_index("ix_studio_artist_memories_product_id", table_name="studio_artist_memories")
    op.drop_table("studio_artist_memories")
    # After the table: the composite FK that needed it is gone.
    op.drop_constraint("uq_studio_artists_id_owner", "studio_artists", type_="unique")
