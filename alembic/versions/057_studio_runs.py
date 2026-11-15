"""Studio planner runs: the ``studio_runs`` table.

Revision ID: 057
Revises: 056
Create Date: 2026-11-14 00:00:00.000000

One table (see ``src/db/models/studio.py::StudioRun`` for the rationale behind each constraint):

* ``studio_runs`` — an LLM job per row: ``queued → running → succeeded | failed``,
  ``queued → cancelled``. ``track_id`` cascades with the track; the result version ids are
  ``ON DELETE SET NULL``.
* ``uq_studio_runs_one_active_per_track`` — partial unique index: one queued/running run per track.
  It, not a prior ``SELECT``, decides ``409 run_in_progress``.
* ``UNIQUE (user_id, idempotency_key)`` — a replayed submit is the same run.
* CHECKs generated from the run enums (kind / status / stage), ``finished_at`` iff terminal, and a
  ``failed`` run always has an ``error_code``.

All three run enums are declared in full now (``plan``, ``lyrics``, ``revise_lyrics``) so the later
lyrics runs need no migration. Downgrade drops the table (and its data).
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "057"
down_revision: str | Sequence[str] | None = "056"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TS = sa.DateTime(timezone=True)
_NOW = sa.text("CURRENT_TIMESTAMP")
_UUID = postgresql.UUID(as_uuid=True)


def upgrade() -> None:
    """Upgrade database schema."""
    op.create_table(
        "studio_runs",
        sa.Column("id", _UUID, nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("user_id", _UUID, nullable=False),
        sa.Column("track_id", _UUID, nullable=False),
        sa.Column("kind", sa.String(20), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("stage", sa.String(20), nullable=True),
        sa.Column("revision", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("idempotency_key", sa.String(128), nullable=False),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("request", postgresql.JSONB(), nullable=False),
        sa.Column("context_snapshot", postgresql.JSONB(), nullable=True),
        sa.Column("prompt_version", sa.String(40), nullable=True),
        sa.Column("model_key", sa.String(60), nullable=False),
        sa.Column("model_id", sa.String(120), nullable=True),
        sa.Column("price_tokens", sa.Integer(), nullable=False),
        sa.Column("billing_account_id", _UUID, nullable=True),
        sa.Column("result_plan_version_id", _UUID, nullable=True),
        sa.Column("result_lyrics_version_id", _UUID, nullable=True),
        sa.Column(
            "warnings",
            postgresql.JSONB(),
            server_default=sa.text("'[]'::jsonb"),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(60), nullable=True),
        sa.Column("error_detail", postgresql.JSONB(), nullable=True),
        sa.Column("llm_calls", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("prompt_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("completion_tokens", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("lease_expires_at", _TS, nullable=True),
        sa.Column("started_at", _TS, nullable=True),
        sa.Column("finished_at", _TS, nullable=True),
        sa.Column("created_at", _TS, server_default=_NOW, nullable=False),
        sa.Column("updated_at", _TS, server_default=_NOW, nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["track_id"], ["studio_tracks.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(
            ["result_plan_version_id"], ["studio_document_versions.id"], ondelete="SET NULL"
        ),
        sa.ForeignKeyConstraint(
            ["result_lyrics_version_id"], ["studio_document_versions.id"], ondelete="SET NULL"
        ),
        sa.UniqueConstraint("user_id", "idempotency_key", name="uq_studio_runs_user_idempotency"),
        sa.CheckConstraint(
            "kind IN ('plan', 'lyrics', 'revise_lyrics')", name="ck_studio_runs_kind"
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_studio_runs_status",
        ),
        sa.CheckConstraint(
            "stage IS NULL OR stage IN ('context', 'drafting', 'validating', 'revising', 'saving')",
            name="ck_studio_runs_stage",
        ),
        sa.CheckConstraint("price_tokens >= 0", name="ck_studio_runs_price"),
        sa.CheckConstraint("revision >= 1", name="ck_studio_runs_revision"),
        sa.CheckConstraint(
            "(status IN ('cancelled', 'failed', 'succeeded')) = (finished_at IS NOT NULL)",
            name="ck_studio_runs_finished",
        ),
        sa.CheckConstraint(
            "status <> 'failed' OR error_code IS NOT NULL", name="ck_studio_runs_failed_code"
        ),
    )
    op.create_index("ix_studio_runs_product_id", "studio_runs", ["product_id"])
    op.create_index(
        "uq_studio_runs_one_active_per_track",
        "studio_runs",
        ["track_id"],
        unique=True,
        postgresql_where=sa.text("status IN ('queued', 'running')"),
    )
    op.create_index(
        "ix_studio_runs_queued",
        "studio_runs",
        ["created_at"],
        postgresql_where=sa.text("status = 'queued'"),
    )
    op.create_index(
        "ix_studio_runs_running_lease",
        "studio_runs",
        ["lease_expires_at"],
        postgresql_where=sa.text("status = 'running'"),
    )
    op.create_index(
        "ix_studio_runs_user_created", "studio_runs", ["user_id", sa.text("created_at DESC")]
    )
    op.create_index(
        "ix_studio_runs_track_created",
        "studio_runs",
        ["track_id", sa.text("created_at DESC"), sa.text("id DESC")],
    )


def downgrade() -> None:
    """Downgrade database schema (drops every planner run and its data)."""
    op.drop_table("studio_runs")
