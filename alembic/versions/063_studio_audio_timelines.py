"""Realized timelines and detail waveform peaks (S3a).

Revision ID: 063
Revises: 062
Create Date: 2026-11-26 12:00:00.000000

One new studio table, two new columns on the audio versions and two on the library's audio profile
(see ``src/db/models/studio.py::StudioAudioTimeline`` and
``src/db/models/library.py::LibraryAudioProfile``):

* ``studio_audio_timelines`` - immutable, numbered full snapshots per **audio version**. The
  composite owner foreign key ``(track_id, user_id, product_id)`` ties a row to its track's owner;
  ``(audio_version_id, track_id)`` ties it to an audio version of that track and **cascades** (a
  timeline means nothing without its audio). The parent key ``(parent_id, audio_version_id)`` is
  ``NO ACTION``, checked at the end of the statement: deleting one parent is refused, while
  deleting the whole audio version removes parents and children together.
  ``UNIQUE NULLS NOT DISTINCT (audio_version_id, parent_id, content_hash)`` makes create
  idempotent (root versions collide too).
* ``studio_track_audio.next_timeline_number`` / ``current_timeline_id`` - the per-audio counter
  (never ``max()+1``) and the current pointer, a composite ``(current_timeline_id, id)`` key
  (``NO ACTION``) so it can only name a timeline of the same audio version.
* ``library_audio_profiles.detail_peaks`` / ``detail_peaks_version`` - the 20 ms peaks level, both
  NULL until computed (new uploads: at ingest; older ones:
  ``python -m src.cli.backfill_audio_peaks``). The column is ``STORAGE EXTERNAL`` (never
  compressed), so ``substring()`` of a window fetches only the TOAST chunks that hold it.

The CHECK, unique and FK names are spelled out here because a migration never imports ``src``.

Downgrade drops everything in reverse order. **It destroys the stored timeline versions** (the
previous code has no use for them). The two profile columns are derived data the backfill can
compute again.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "063"
down_revision: str | Sequence[str] | None = "062"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TS = sa.DateTime(timezone=True)
_NOW = sa.text("CURRENT_TIMESTAMP")
_UUID = postgresql.UUID(as_uuid=True)

# Frozen literals (a migration never imports ``src``). One int8 min and one int8 max per 20 ms
# bucket, for the longest accepted audio (``AUDIO_MAX_SECONDS`` = 900) plus the one second of slack
# the peaks decoder is cut at: 901 s x 50 buckets/s x 2 bytes = 90,100 bytes.
_AUDIO_MAX_SECONDS = 900
_DETAIL_BUCKETS_PER_SECOND = 50
_MAX_DETAIL_PEAKS_BYTES = (_AUDIO_MAX_SECONDS + 1) * _DETAIL_BUCKETS_PER_SECOND * 2


def upgrade() -> None:
    """Upgrade database schema."""
    # -- studio_audio_timelines ----------------------------------------------------------------
    op.create_table(
        "studio_audio_timelines",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("track_id", _UUID, nullable=False),
        sa.Column("audio_version_id", _UUID, nullable=False),
        sa.Column("version_number", sa.Integer(), nullable=False),
        sa.Column("parent_id", _UUID, nullable=True),
        sa.Column("origin", sa.String(10), nullable=False),
        sa.Column("schema_version", sa.SmallInteger(), nullable=False),
        sa.Column("content", postgresql.JSONB(), nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False),
        sa.Column("is_complete", sa.Boolean(), nullable=False),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["track_id", "user_id", "product_id"],
            ["studio_tracks.id", "studio_tracks.user_id", "studio_tracks.product_id"],
            name="fk_studio_audio_timelines_track_owner",
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["audio_version_id", "track_id"],
            ["studio_track_audio.id", "studio_track_audio.track_id"],
            name="fk_studio_audio_timelines_audio",
            ondelete="CASCADE",
        ),
        # The target of the parent key below and of the audio version's current pointer.
        sa.UniqueConstraint("id", "audio_version_id", name="uq_studio_audio_timelines_id_audio"),
        sa.ForeignKeyConstraint(
            ["parent_id", "audio_version_id"],
            ["studio_audio_timelines.id", "studio_audio_timelines.audio_version_id"],
            name="fk_studio_audio_timelines_parent",
        ),
        sa.UniqueConstraint(
            "audio_version_id", "version_number", name="uq_studio_audio_timelines_number"
        ),
        sa.UniqueConstraint(
            "audio_version_id",
            "parent_id",
            "content_hash",
            name="uq_studio_audio_timelines_content",
            postgresql_nulls_not_distinct=True,
        ),
        sa.CheckConstraint("version_number >= 1", name="ck_studio_audio_timelines_number"),
        sa.CheckConstraint(
            "origin IN ('user', 'aligner')", name="ck_studio_audio_timelines_origin"
        ),
    )
    op.create_index(
        "ix_studio_audio_timelines_product_id", "studio_audio_timelines", ["product_id"]
    )
    # The users and studio_tracks cascades find their rows by these.
    op.create_index("ix_studio_audio_timelines_user_id", "studio_audio_timelines", ["user_id"])
    op.create_index("ix_studio_audio_timelines_track_id", "studio_audio_timelines", ["track_id"])
    op.create_index(
        "ix_studio_audio_timelines_list",
        "studio_audio_timelines",
        ["audio_version_id", sa.text("version_number DESC")],
    )

    # -- the audio version: the counter and the current pointer --------------------------------
    op.add_column(
        "studio_track_audio",
        sa.Column(
            "next_timeline_number", sa.Integer(), nullable=False, server_default=sa.text("1")
        ),
    )
    op.add_column("studio_track_audio", sa.Column("current_timeline_id", _UUID, nullable=True))
    op.create_foreign_key(
        "fk_studio_track_audio_current_timeline",
        "studio_track_audio",
        "studio_audio_timelines",
        ["current_timeline_id", "id"],
        ["id", "audio_version_id"],
    )

    # -- the library's detail peaks level ------------------------------------------------------
    op.add_column(
        "library_audio_profiles", sa.Column("detail_peaks", sa.LargeBinary(), nullable=True)
    )
    op.add_column(
        "library_audio_profiles",
        sa.Column("detail_peaks_version", sa.SmallInteger(), nullable=True),
    )
    # Uncompressed out-of-line storage: a window read fetches only the chunks it needs.
    op.execute(
        sa.text("ALTER TABLE library_audio_profiles ALTER COLUMN detail_peaks SET STORAGE EXTERNAL")
    )
    op.create_check_constraint(
        "ck_library_audio_profiles_detail_peaks_pair",
        "library_audio_profiles",
        "(detail_peaks IS NULL) = (detail_peaks_version IS NULL)",
    )
    op.create_check_constraint(
        "ck_library_audio_profiles_detail_peaks_size",
        "library_audio_profiles",
        "detail_peaks IS NULL OR (octet_length(detail_peaks) % 2 = 0 "
        f"AND octet_length(detail_peaks) <= {_MAX_DETAIL_PEAKS_BYTES})",
    )


def downgrade() -> None:
    """Downgrade database schema (drops every stored timeline version)."""
    op.drop_constraint(
        "ck_library_audio_profiles_detail_peaks_size", "library_audio_profiles", type_="check"
    )
    op.drop_constraint(
        "ck_library_audio_profiles_detail_peaks_pair", "library_audio_profiles", type_="check"
    )
    op.drop_column("library_audio_profiles", "detail_peaks_version")
    op.drop_column("library_audio_profiles", "detail_peaks")
    # The pointer references the timelines, so it goes before their table.
    op.drop_constraint(
        "fk_studio_track_audio_current_timeline", "studio_track_audio", type_="foreignkey"
    )
    op.drop_column("studio_track_audio", "current_timeline_id")
    op.drop_column("studio_track_audio", "next_timeline_number")
    op.drop_table("studio_audio_timelines")
