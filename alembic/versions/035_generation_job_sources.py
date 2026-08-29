"""Add ordered generation source provenance.

Revision ID: 035
Revises: 034
Create Date: 2026-08-29 00:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import context, op

revision: str = "035"
down_revision: str | Sequence[str] | None = "034"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_BATCH_SIZE = 10_000


def _backfill_batch(job_ids: list[object]) -> None:
    """Insert legacy primary source data for one bounded set of jobs."""
    bind = sa.bindparam("job_ids", expanding=True)
    output_sql = sa.text(
        """
        INSERT INTO generation_job_sources
            (job_id, position, product_id, source_upload_id, source_output_id, asset_ref, media_kind)
        SELECT j.id, 0, j.product_id, NULL, j.source_output_id,
               'output:' || j.source_output_id::text,
               -- Keep this in sync with media_kind_from_content_type: unknown
               -- families must not be silently classified as images.
               CASE
                   WHEN o.content_type LIKE 'video/%' THEN 'video'
                   WHEN o.content_type LIKE 'image/%' THEN 'image'
                   ELSE NULL
               END
        FROM generation_jobs j
        JOIN generation_outputs o ON o.id = j.source_output_id
        WHERE j.id IN :job_ids AND j.source_output_id IS NOT NULL
        ON CONFLICT (job_id, position) DO NOTHING
        """
    ).bindparams(bind)
    upload_sql = sa.text(
        """
        WITH legacy_uploads AS (
            SELECT j.id AS job_id, j.product_id, j.input_image_id AS upload_id,
                   (j.source_output_id IS NOT NULL)::int AS position_offset
            FROM generation_jobs j
            WHERE j.id IN :job_ids AND j.input_image_id IS NOT NULL
            UNION
            SELECT j.id AS job_id, j.product_id, o.input_image_id AS upload_id,
                   (j.source_output_id IS NOT NULL)::int AS position_offset
            FROM generation_jobs j
            JOIN generation_outputs o ON o.job_id = j.id
            WHERE j.id IN :job_ids AND o.input_image_id IS NOT NULL
        ), deduplicated AS (
            SELECT DISTINCT job_id, product_id, upload_id, position_offset
            FROM legacy_uploads
        ), numbered AS (
            SELECT job_id, product_id, upload_id, position_offset,
                   row_number() OVER (PARTITION BY job_id ORDER BY upload_id) - 1 AS upload_position
            FROM deduplicated
        )
        INSERT INTO generation_job_sources
            (job_id, position, product_id, source_upload_id, source_output_id, asset_ref, media_kind)
        SELECT n.job_id, n.position_offset + n.upload_position, n.product_id, n.upload_id, NULL,
               'upload:' || n.upload_id::text,
               -- Mirrors media_kind_from_content_type; unknown families are
               -- retained as an explicitly unclassified historical source.
               CASE
                   WHEN u.content_type LIKE 'video/%' THEN 'video'
                   WHEN u.content_type LIKE 'image/%' THEN 'image'
                   ELSE NULL
               END
        FROM numbered n
        JOIN user_images u ON u.id = n.upload_id
        ON CONFLICT (job_id, position) DO NOTHING
        """
    ).bindparams(bind)
    op.get_bind().execute(output_sql, {"job_ids": job_ids})
    op.get_bind().execute(upload_sql, {"job_ids": job_ids})


def _backfill() -> None:
    """Keyset-batch historical source rows to keep migration locks bounded."""
    bind = op.get_bind()
    last_id: object | None = None
    while True:
        where = "" if last_id is None else "WHERE id > :last_id"
        ids = list(
            bind.execute(
                sa.text(
                    f"SELECT id FROM generation_jobs {where} ORDER BY id LIMIT :batch_size"  # noqa: S608
                ),
                {"last_id": last_id, "batch_size": _BATCH_SIZE},
            ).scalars()
        )
        if not ids:
            return
        _backfill_batch(ids)
        last_id = ids[-1]


def upgrade() -> None:
    """Create the ordered provenance table and backfill legacy job rows."""
    op.create_table(
        "generation_job_sources",
        sa.Column("job_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False),
        sa.Column("product_id", sa.String(length=32), nullable=False),
        sa.Column("source_upload_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_output_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("asset_ref", sa.String(length=64), nullable=False),
        sa.Column("media_kind", sa.String(length=16), nullable=True),
        sa.CheckConstraint(
            "NOT (source_upload_id IS NOT NULL AND source_output_id IS NOT NULL)",
            name="ck_generation_job_sources_single_source",
        ),
        sa.ForeignKeyConstraint(["job_id"], ["generation_jobs.id"], ondelete="CASCADE"),
        sa.ForeignKeyConstraint(["source_upload_id"], ["user_images.id"], ondelete="SET NULL"),
        sa.ForeignKeyConstraint(
            ["source_output_id"],
            ["generation_outputs.id"],
            name="fk_generation_job_sources_source_output_id",
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("job_id", "position"),
    )
    op.create_index(
        "ix_generation_job_sources_product_job",
        "generation_job_sources",
        ["product_id", "job_id"],
    )
    op.create_index(
        "ix_generation_job_sources_source_upload_id",
        "generation_job_sources",
        ["source_upload_id"],
    )
    op.create_index(
        "ix_generation_job_sources_source_output_id",
        "generation_job_sources",
        ["source_output_id"],
    )
    # Offline SQL rendering cannot inspect keyset batches. The generated DDL
    # remains useful for review; production upgrades run online and perform
    # the bounded backfill in the same migration transaction.
    if not context.is_offline_mode():
        _backfill()


def downgrade() -> None:
    """Drop ordered source provenance; legacy columns remain untouched."""
    op.drop_table("generation_job_sources")
