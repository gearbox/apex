"""Media Project Studio foundation: artists, project profiles, tracks, document versions.

Revision ID: 055
Revises: 054
Create Date: 2026-11-11 00:00:00.000000

Four tables (see ``src/db/models/studio.py`` for the rationale behind each constraint):

* ``studio_artists`` — reusable artist personas; case-insensitive unique name per owner.
* ``studio_project_profiles`` — 1:1 studio extension of ``library_projects`` (PK is the library
  project id, ``ON DELETE CASCADE``).
* ``studio_tracks`` — ordered tracks; ``(project, position)`` is a *deferred* unique so a reorder
  can swap positions inside one statement.
* ``studio_document_versions`` — immutable plan/lyrics snapshots in a per-(track, kind) tree.

``studio_tracks`` and ``studio_document_versions`` reference each other (current pointers vs
``track_id``), so the two ``current_*`` composite FKs are created last, after both tables exist.
Downgrade drops them first and then the tables in reverse order.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "055"
down_revision: str | Sequence[str] | None = "054"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TS = sa.DateTime(timezone=True)
_NOW = sa.text("CURRENT_TIMESTAMP")


def upgrade() -> None:
    """Upgrade database schema."""
    # -- studio_artists ---------------------------------------------------------------------
    op.create_table(
        "studio_artists",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("persona", sa.Text(), nullable=True),
        sa.Column("voice_notes", sa.Text(), nullable=True),
        sa.Column("default_language", sa.String(35), nullable=True),
        sa.Column(
            "aesthetic_references",
            postgresql.ARRAY(sa.Text()),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "style_tags",
            postgresql.ARRAY(sa.String(60)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column(
            "global_avoid",
            postgresql.ARRAY(sa.String(60)),
            server_default=sa.text("'{}'"),
            nullable=False,
        ),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
    )
    op.create_index("ix_studio_artists_product_id", "studio_artists", ["product_id"])
    op.create_index(
        "uq_studio_artists_owner_name",
        "studio_artists",
        ["product_id", "user_id", sa.text("lower(name)")],
        unique=True,
    )
    op.create_index(
        "ix_studio_artists_owner_created",
        "studio_artists",
        ["user_id", "product_id", "created_at", "id"],
    )

    # -- studio_project_profiles -----------------------------------------------------------
    op.create_table(
        "studio_project_profiles",
        sa.Column("library_project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("artist_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("release_kind", sa.String(10), nullable=False),
        sa.Column("concept", sa.Text(), nullable=True),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("library_project_id"),
        sa.ForeignKeyConstraint(
            ["library_project_id"], ["library_projects.id"], ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # NO ACTION on purpose: the backstop behind 409 artist_in_use.
        sa.ForeignKeyConstraint(["artist_id"], ["studio_artists.id"]),
        sa.CheckConstraint(
            "release_kind IN ('single', 'ep', 'album')",
            name="ck_studio_project_profiles_release_kind",
        ),
    )
    op.create_index(
        "ix_studio_project_profiles_product_id", "studio_project_profiles", ["product_id"]
    )
    op.create_index(
        "ix_studio_project_profiles_artist_id", "studio_project_profiles", ["artist_id"]
    )
    op.create_index(
        "ix_studio_project_profiles_owner_created",
        "studio_project_profiles",
        ["user_id", "product_id", "created_at", "library_project_id"],
    )

    # -- studio_tracks (without the circular current_* FKs) -----------------------------------
    op.create_table(
        "studio_tracks",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("library_project_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column(
            "next_plan_version_number", sa.Integer(), server_default=sa.text("1"), nullable=False
        ),
        sa.Column(
            "next_lyrics_version_number", sa.Integer(), server_default=sa.text("1"), nullable=False
        ),
        sa.Column("current_plan_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("current_lyrics_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        # NO ACTION (not RESTRICT): checked at statement end, so the user hard-delete cascade
        # does not depend on the order in which the cascading deletes run.
        sa.ForeignKeyConstraint(
            ["library_project_id"], ["studio_project_profiles.library_project_id"]
        ),
        sa.UniqueConstraint(
            "library_project_id",
            "position",
            name="uq_studio_tracks_project_position",
            deferrable=True,
            initially="DEFERRED",
        ),
        sa.CheckConstraint("position >= 0", name="ck_studio_tracks_position"),
        sa.CheckConstraint(
            "current_lyrics_version_id IS NULL OR current_plan_version_id IS NOT NULL",
            name="ck_studio_tracks_lyrics_need_plan",
        ),
    )
    op.create_index("ix_studio_tracks_product_id", "studio_tracks", ["product_id"])

    # -- studio_document_versions -------------------------------------------------------------
    op.create_table(
        "studio_document_versions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("track_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("kind", sa.String(10), nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("parent_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("based_on_plan_version_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("schema_version", sa.SmallInteger(), nullable=False),
        sa.Column("origin", sa.String(10), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["track_id"], ["studio_tracks.id"], ondelete="CASCADE"),
        sa.UniqueConstraint(
            "track_id", "kind", "version_number", name="uq_studio_doc_versions_number"
        ),
        sa.UniqueConstraint("id", "track_id", name="uq_studio_doc_versions_id_track"),
        sa.UniqueConstraint("id", "track_id", "kind", name="uq_studio_doc_versions_id_track_kind"),
        sa.UniqueConstraint(
            "track_id",
            "kind",
            "parent_version_id",
            "content_hash",
            name="uq_studio_doc_versions_sibling_content",
            postgresql_nulls_not_distinct=True,
        ),
        sa.ForeignKeyConstraint(
            ["parent_version_id", "track_id", "kind"],
            [
                "studio_document_versions.id",
                "studio_document_versions.track_id",
                "studio_document_versions.kind",
            ],
            name="fk_studio_doc_versions_parent",
        ),
        sa.ForeignKeyConstraint(
            ["based_on_plan_version_id", "track_id"],
            ["studio_document_versions.id", "studio_document_versions.track_id"],
            name="fk_studio_doc_versions_based_on_plan",
        ),
        sa.CheckConstraint("kind IN ('plan', 'lyrics')", name="ck_studio_doc_versions_kind"),
        sa.CheckConstraint(
            "origin IN ('user', 'llm', 'import')", name="ck_studio_doc_versions_origin"
        ),
        sa.CheckConstraint("version_number >= 1", name="ck_studio_doc_versions_number"),
        sa.CheckConstraint("schema_version >= 1", name="ck_studio_doc_versions_schema_version"),
        sa.CheckConstraint(
            "(kind = 'lyrics') = (based_on_plan_version_id IS NOT NULL)",
            name="ck_studio_doc_versions_based_on_iff_lyrics",
        ),
    )
    op.create_index(
        "ix_studio_document_versions_product_id", "studio_document_versions", ["product_id"]
    )
    op.create_index(
        "ix_studio_doc_versions_based_on", "studio_document_versions", ["based_on_plan_version_id"]
    )
    op.create_index(
        "ix_studio_doc_versions_list",
        "studio_document_versions",
        ["track_id", "kind", sa.text("created_at DESC"), sa.text("id DESC")],
    )

    # -- the circular pointers, last ----------------------------------------------------------
    op.create_foreign_key(
        "fk_studio_tracks_current_plan_version",
        "studio_tracks",
        "studio_document_versions",
        ["current_plan_version_id", "id"],
        ["id", "track_id"],
    )
    op.create_foreign_key(
        "fk_studio_tracks_current_lyrics_version",
        "studio_tracks",
        "studio_document_versions",
        ["current_lyrics_version_id", "id"],
        ["id", "track_id"],
    )


def downgrade() -> None:
    """Downgrade database schema (drops every studio table and its data)."""
    op.drop_constraint(
        "fk_studio_tracks_current_lyrics_version", "studio_tracks", type_="foreignkey"
    )
    op.drop_constraint("fk_studio_tracks_current_plan_version", "studio_tracks", type_="foreignkey")

    op.drop_index("ix_studio_doc_versions_list", table_name="studio_document_versions")
    op.drop_index("ix_studio_doc_versions_based_on", table_name="studio_document_versions")
    op.drop_index("ix_studio_document_versions_product_id", table_name="studio_document_versions")
    op.drop_table("studio_document_versions")

    op.drop_index("ix_studio_tracks_product_id", table_name="studio_tracks")
    op.drop_table("studio_tracks")

    op.drop_index("ix_studio_project_profiles_owner_created", table_name="studio_project_profiles")
    op.drop_index("ix_studio_project_profiles_artist_id", table_name="studio_project_profiles")
    op.drop_index("ix_studio_project_profiles_product_id", table_name="studio_project_profiles")
    op.drop_table("studio_project_profiles")

    op.drop_index("ix_studio_artists_owner_created", table_name="studio_artists")
    op.drop_index("uq_studio_artists_owner_name", table_name="studio_artists")
    op.drop_index("ix_studio_artists_product_id", table_name="studio_artists")
    op.drop_table("studio_artists")
