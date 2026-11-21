"""Studio artist memories: the ``studio_artist_memories`` table.

Revision ID: 059
Revises: 058
Create Date: 2026-11-20 00:00:00.000000

One table (see ``src/db/models/studio.py::StudioArtistMemory`` for the rationale):

* ``studio_artist_memories`` - episodes of an artist's fictional life that planner runs use as
  inspiration. ``artist_id`` is ``ON DELETE CASCADE``: ownership and product scoping are inherited
  through the artist (P2-M1), so the table has no ``user_id`` / ``product_id`` of its own, and a
  user hard-delete reaches it through ``studio_artists``.
* CHECKs for ``salience`` (1-5) and for ``status`` / ``origin`` (generated from the enums in the
  model; spelled out here because a migration never imports ``src``). ``origin`` is declared in
  full now (``user``, ``chat``) so the artist chat (P2c) needs no migration.
* ``ix_studio_artist_memories_list`` serves the cursor list (newest first, per artist);
  ``ix_studio_artist_memories_selection`` is a partial index over active rows in the selection
  order (``salience DESC, occurred_on DESC NULLS LAST, id``).

``themes`` / ``imagery`` are varchar arrays, which PostgreSQL truncates silently on the bind cast:
the service's length check (after NFC) is the only guard.

Downgrade drops the table (and every memory in it).
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
    op.create_table(
        "studio_artist_memories",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
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
        sa.ForeignKeyConstraint(["artist_id"], ["studio_artists.id"], ondelete="CASCADE"),
        sa.CheckConstraint("salience BETWEEN 1 AND 5", name="ck_studio_artist_memories_salience"),
        sa.CheckConstraint(
            "status IN ('active', 'archived')", name="ck_studio_artist_memories_status"
        ),
        sa.CheckConstraint("origin IN ('user', 'chat')", name="ck_studio_artist_memories_origin"),
    )
    op.create_index(
        "ix_studio_artist_memories_list",
        "studio_artist_memories",
        ["artist_id", sa.text("created_at DESC"), sa.text("id DESC")],
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
    op.drop_table("studio_artist_memories")
