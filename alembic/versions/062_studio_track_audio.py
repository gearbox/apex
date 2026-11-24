"""Studio track audio and library audio profiles (S2a).

Revision ID: 062
Revises: 061
Create Date: 2026-11-24 12:00:00.000000

One new library table, one new studio table and two new track columns (see
``src/db/models/library.py::LibraryAudioProfile`` and ``src/db/models/studio.py::StudioTrackAudio``
for the rationale):

* ``library_audio_profiles`` - the probed facts and waveform peaks of one stored audio asset,
  library-owned (D12) and 1:1 with its asset. Exactly one of ``upload_id`` / ``output_id`` is set
  (CHECK), each a typed foreign key with ``ON DELETE CASCADE`` (the profile has no life of its own:
  a delete, the retention sweep and a user hard-delete remove it with the asset). The ``output_id``
  arm is unused until generated audio exists (S2b). ``peaks`` is exactly 2,048 bytes (CHECK).
  A partial unique index per arm makes "two profiles for one asset" unrepresentable.
* ``studio_tracks.uq_studio_tracks_id_owner`` - ``(id, user_id, product_id)``, the target of the
  audio versions' composite owner foreign key (the same device P2b used for artists).
* ``studio_track_audio`` - a numbered audio version of a track. ``hold_id`` is ``NOT NULL UNIQUE``
  (FK to ``library_asset_holds``, ``NO ACTION``): "every audio version is pinned" is a database
  invariant, and the asset is read through the hold (no second copy of it, hence no
  ``asset_ref`` column). The optional ``plan_version_id`` / ``lyrics_version_id`` are composite
  ``(version, track_id)`` foreign keys to ``studio_document_versions`` (``NO ACTION``), so another
  track's version is unrepresentable and deleting a realized version is refused.
* ``studio_tracks.next_audio_version_number`` and ``studio_tracks.current_audio_version_id`` - the
  per-track counter (never ``max()+1``) and the current-audio pointer, a composite
  ``(current_audio_version_id, id)`` foreign key to ``studio_track_audio (id, track_id)``
  (``NO ACTION``, like the plan / lyrics pointers).

The CHECK, unique and FK names are spelled out here because a migration never imports ``src``.

Downgrade drops the track columns and then the audio versions and profiles (the holds the audio
versions took are removed too, so no pin outlives the revision; the assets themselves are untouched
and their retention clock is restarted, the same as a last release).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "062"
down_revision: str | Sequence[str] | None = "061"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TS = sa.DateTime(timezone=True)
_NOW = sa.text("CURRENT_TIMESTAMP")
_UUID = postgresql.UUID(as_uuid=True)

_HOLDER_KIND = "studio.track_audio"


def upgrade() -> None:
    """Upgrade database schema."""
    # -- library_audio_profiles ----------------------------------------------------------------
    op.create_table(
        "library_audio_profiles",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("upload_id", _UUID, nullable=True),
        sa.Column("output_id", _UUID, nullable=True),
        sa.Column("codec", sa.String(16), nullable=False),
        sa.Column("sample_rate_hz", sa.Integer(), nullable=False),
        sa.Column("channels", sa.SmallInteger(), nullable=False),
        sa.Column("bit_rate", sa.Integer(), nullable=True),
        sa.Column("peaks", sa.LargeBinary(), nullable=False),
        sa.Column("peaks_version", sa.SmallInteger(), nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["upload_id"], ["user_images.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["output_id"], ["generation_outputs.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "num_nonnulls(upload_id, output_id) = 1", name="ck_library_audio_profiles_one_asset"
        ),
        sa.CheckConstraint(
            "octet_length(peaks) = 2048", name="ck_library_audio_profiles_peaks_size"
        ),
        sa.CheckConstraint("channels BETWEEN 1 AND 8", name="ck_library_audio_profiles_channels"),
    )
    op.create_index(
        "uq_library_audio_profiles_upload",
        "library_audio_profiles",
        ["upload_id"],
        unique=True,
        postgresql_where=sa.text("upload_id IS NOT NULL"),
    )
    op.create_index(
        "uq_library_audio_profiles_output",
        "library_audio_profiles",
        ["output_id"],
        unique=True,
        postgresql_where=sa.text("output_id IS NOT NULL"),
    )

    # -- the track: the composite-FK target, the counter and the pointer column ----------------
    op.create_unique_constraint(
        "uq_studio_tracks_id_owner", "studio_tracks", ["id", "user_id", "product_id"]
    )
    op.add_column(
        "studio_tracks",
        sa.Column(
            "next_audio_version_number", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
    )
    op.add_column("studio_tracks", sa.Column("current_audio_version_id", _UUID, nullable=True))

    # -- studio_track_audio --------------------------------------------------------------------
    op.create_table(
        "studio_track_audio",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("track_id", _UUID, nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("label", sa.String(60), nullable=True),
        sa.Column("origin", sa.String(10), nullable=False),
        sa.Column("hold_id", _UUID, nullable=False),
        sa.Column("plan_version_id", _UUID, nullable=True),
        sa.Column("lyrics_version_id", _UUID, nullable=True),
        sa.Column("duration_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["hold_id"],
            ["library_asset_holds.id"],
            name="fk_studio_track_audio_hold",
            ondelete="NO ACTION",
        ),
        sa.ForeignKeyConstraint(
            ["track_id", "user_id", "product_id"],
            ["studio_tracks.id", "studio_tracks.user_id", "studio_tracks.product_id"],
            name="fk_studio_track_audio_track_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["plan_version_id", "track_id"],
            ["studio_document_versions.id", "studio_document_versions.track_id"],
            name="fk_studio_track_audio_plan_version",
        ),
        sa.ForeignKeyConstraint(
            ["lyrics_version_id", "track_id"],
            ["studio_document_versions.id", "studio_document_versions.track_id"],
            name="fk_studio_track_audio_lyrics_version",
        ),
        sa.UniqueConstraint("hold_id", name="studio_track_audio_hold_id_key"),
        sa.UniqueConstraint("track_id", "version_number", name="uq_studio_track_audio_number"),
        sa.UniqueConstraint("id", "track_id", name="uq_studio_track_audio_id_track"),
        sa.CheckConstraint("version_number >= 1", name="ck_studio_track_audio_number"),
        sa.CheckConstraint("duration_ms > 0", name="ck_studio_track_audio_duration"),
        sa.CheckConstraint("origin IN ('upload')", name="ck_studio_track_audio_origin"),
    )
    op.create_index("ix_studio_track_audio_product_id", "studio_track_audio", ["product_id"])
    op.create_index("ix_studio_track_audio_user_id", "studio_track_audio", ["user_id"])
    op.create_index(
        "ix_studio_track_audio_plan_version",
        "studio_track_audio",
        ["plan_version_id"],
        postgresql_where=sa.text("plan_version_id IS NOT NULL"),
    )
    op.create_index(
        "ix_studio_track_audio_lyrics_version",
        "studio_track_audio",
        ["lyrics_version_id"],
        postgresql_where=sa.text("lyrics_version_id IS NOT NULL"),
    )

    # -- the track's current-audio pointer (after the table it points at exists) ---------------
    op.create_foreign_key(
        "fk_studio_tracks_current_audio_version",
        "studio_tracks",
        "studio_track_audio",
        ["current_audio_version_id", "id"],
        ["id", "track_id"],
    )


def downgrade() -> None:
    """Downgrade database schema (drops every audio version and audio profile)."""
    # Reverse order: the track pointer references the audio versions, which reference holds.
    op.drop_constraint(
        "fk_studio_tracks_current_audio_version", "studio_tracks", type_="foreignkey"
    )
    # The holds the audio versions took go with them (no pin may outlive the revision), and the
    # assets they pinned get their retention clock restarted, as a last release does: otherwise a
    # long-pinned upload (its ``expires_at`` far in the past) would be swept the moment the pin
    # goes. ``greatest`` never shortens a clock. 7 days is the default ``RETENTION_DAYS``: a
    # migration cannot read settings.
    op.execute(
        sa.text(
            """
            UPDATE user_images SET expires_at = greatest(expires_at, now() + interval '7 days')
            WHERE id IN (
                SELECT upload_id FROM library_asset_holds
                WHERE holder_kind = :kind AND upload_id IS NOT NULL
            )
            """
        ).bindparams(kind=_HOLDER_KIND)
    )
    op.execute(
        sa.text(
            """
            UPDATE generation_outputs SET expires_at = greatest(expires_at, now() + interval '7 days')
            WHERE id IN (
                SELECT output_id FROM library_asset_holds
                WHERE holder_kind = :kind AND output_id IS NOT NULL
            )
            """
        ).bindparams(kind=_HOLDER_KIND)
    )
    # Audio rows first (they reference their holds), then the holds.
    op.drop_table("studio_track_audio")
    op.execute(
        sa.text("DELETE FROM library_asset_holds WHERE holder_kind = :kind").bindparams(
            kind=_HOLDER_KIND
        )
    )
    op.drop_column("studio_tracks", "current_audio_version_id")
    op.drop_column("studio_tracks", "next_audio_version_number")
    op.drop_constraint("uq_studio_tracks_id_owner", "studio_tracks", type_="unique")
    op.drop_table("library_audio_profiles")
