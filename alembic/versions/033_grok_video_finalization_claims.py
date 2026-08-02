"""Add normalized failures, materialization guards, and auth-failure preferences.

Revision ID: 033
Revises: 032
Create Date: 2026-08-01 00:00:00.000000

This squashes the unmerged 033--035 branch migrations.  ``failure_code`` and
``public_error_message`` let API/SSE consumers avoid parsing legacy error
diagnostics; existing ``error_message`` values are intentionally not copied.
Legacy thumbnail rows with an incomplete canonical key are inventory-only:
they are never folded into a global duplicate group.  Rows which *can* be
proved to be duplicates are remapped before deletion and their R2 keys are
put in a durable outbox, processed only after this transaction commits.

The dedicated provider-authentication notification class is seeded for every
admin who already has a Telegram link, with an independent default throttle.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "033"
down_revision: str | Sequence[str] | None = "032"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_NOTIFICATION_CLASS = "provider_authentication.failed"


def _populate_duplicate_map(*, thumbnails: bool) -> None:
    """Populate map without grouping NULL legacy thumbnail keys together."""
    if thumbnails:
        op.execute(
            sa.text(
                """
                INSERT INTO _generation_output_duplicate_map (duplicate_id, canonical_id)
                SELECT id, canonical_id
                FROM (
                    SELECT id,
                           first_value(id) OVER duplicate_window AS canonical_id,
                           row_number() OVER duplicate_window AS duplicate_rank
                    FROM generation_outputs
                    WHERE is_thumbnail = TRUE
                      AND parent_output_id IS NOT NULL
                      AND thumbnail_max_edge IS NOT NULL
                    WINDOW duplicate_window AS (
                        PARTITION BY parent_output_id, thumbnail_max_edge
                        ORDER BY created_at ASC, id ASC
                    )
                ) ranked
                WHERE duplicate_rank > 1
                """
            )
        )
    else:
        op.execute(
            sa.text(
                """
                INSERT INTO _generation_output_duplicate_map (duplicate_id, canonical_id)
                SELECT id, canonical_id
                FROM (
                    SELECT id,
                           first_value(id) OVER duplicate_window AS canonical_id,
                           row_number() OVER duplicate_window AS duplicate_rank
                    FROM generation_outputs
                    WHERE is_thumbnail = FALSE
                    WINDOW duplicate_window AS (
                        PARTITION BY job_id, output_index
                        ORDER BY created_at ASC, id ASC
                    )
                ) ranked
                WHERE duplicate_rank > 1
                """
            )
        )


def _preflight_and_remap_duplicates() -> None:
    """Move all preservable references, queue keys, then delete mapped rows.

    Every metadata/tag row is first resolved to its final canonical output.
    Metadata has a one-row-per-asset uniqueness rule, so a non-lossless
    metadata collision aborts before mutation. Tags are set memberships and
    are instead remapped with an INSERT DISTINCT / DELETE sequence.
    """
    # asyncpg prepares each statement individually, so keep the DROP and
    # CREATE separate rather than sending a multi-statement string.
    op.execute("DROP TABLE IF EXISTS _generation_output_reference_targets")
    op.execute(
        """
        CREATE TEMP TABLE _generation_output_reference_targets ON COMMIT DROP AS
        SELECT remap.source_output_id,
               remap.canonical_output_id,
               canonical.product_id,
               canonical.user_id
        FROM (
            SELECT duplicate_id AS source_output_id,
                   canonical_id AS canonical_output_id
            FROM _generation_output_duplicate_map
            UNION
            SELECT canonical_id AS source_output_id,
                   canonical_id AS canonical_output_id
            FROM _generation_output_duplicate_map
        ) remap
        JOIN generation_outputs canonical ON canonical.id = remap.canonical_output_id
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1
                FROM library_asset_metadata metadata
                JOIN _generation_output_reference_targets target
                  ON target.source_output_id = metadata.asset_id
                WHERE metadata.asset_type = 'output'
                  AND (metadata.product_id <> target.product_id
                       OR metadata.user_id <> target.user_id)
            ) THEN
                RAISE EXCEPTION
                    '033 preflight refused cross-scope output metadata; '
                    'repair metadata product/user ownership before retrying';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM library_asset_metadata metadata
                JOIN _generation_output_reference_targets target
                  ON target.source_output_id = metadata.asset_id
                WHERE metadata.asset_type = 'output'
                GROUP BY metadata.product_id,
                         metadata.user_id,
                         metadata.asset_type,
                         target.canonical_output_id
                HAVING count(*) > 1
            ) THEN
                RAISE EXCEPTION
                    '033 preflight refused ambiguous output duplicate metadata; '
                    'run the output-duplicate repair command before retrying';
            END IF;

            IF EXISTS (
                SELECT 1
                FROM library_asset_tags tag
                JOIN _generation_output_reference_targets target
                  ON target.source_output_id = tag.asset_id
                JOIN library_tags owner ON owner.id = tag.tag_id
                WHERE tag.asset_type = 'output'
                  AND (tag.product_id <> target.product_id
                       OR tag.user_id <> target.user_id
                       OR owner.product_id <> target.product_id
                       OR owner.user_id <> target.user_id)
            ) THEN
                RAISE EXCEPTION
                    '033 preflight refused cross-scope output tag membership; '
                    'repair tag product/user ownership before retrying';
            END IF;
        END $$;
        """
    )

    # Queue physical cleanup in the same transaction as logical deletion.
    # The worker never sees these records unless this migration commits.
    op.execute(
        """
        INSERT INTO storage_cleanup_records (product_id, storage_key, reason)
        SELECT output.product_id, output.storage_key, 'migration_033_duplicate_output'
        FROM generation_outputs output
        JOIN _generation_output_duplicate_map map ON map.duplicate_id = output.id
        ON CONFLICT (storage_key) DO NOTHING
        """
    )

    # Relationships with a real FK are remapped before deletion so their
    # on-delete actions cannot erase source lineage or derivative records.
    for statement in (
        """
        UPDATE generation_outputs output
           SET parent_output_id = map.canonical_id
          FROM _generation_output_duplicate_map map
         WHERE output.parent_output_id = map.duplicate_id
        """,
        """
        UPDATE generation_jobs job
           SET source_output_id = map.canonical_id
          FROM _generation_output_duplicate_map map
         WHERE job.source_output_id = map.duplicate_id
        """,
        """
        UPDATE user_images image
           SET source_output_id = map.canonical_id
          FROM _generation_output_duplicate_map map
         WHERE image.source_output_id = map.duplicate_id
        """,
        """
        UPDATE frame_extraction_jobs frame_job
           SET source_output_id = map.canonical_id
          FROM _generation_output_duplicate_map map
         WHERE frame_job.source_output_id = map.duplicate_id
        """,
        """
        UPDATE library_asset_metadata metadata
           SET asset_id = target.canonical_output_id
          FROM _generation_output_reference_targets target
         WHERE metadata.asset_type = 'output'
           AND metadata.asset_id = target.source_output_id
           AND target.source_output_id <> target.canonical_output_id
        """,
    ):
        op.execute(statement)

    # Tag membership is set-valued. Insert every final membership first,
    # tolerate existing canonical memberships, then remove only source rows.
    # A bulk UPDATE would collide when two-or-more duplicates share a tag.
    op.execute(
        """
        INSERT INTO library_asset_tags (
            tag_id, asset_type, asset_id, product_id, user_id, created_at
        )
        SELECT DISTINCT tag.tag_id,
               'output',
               target.canonical_output_id,
               tag.product_id,
               tag.user_id,
               tag.created_at
        FROM library_asset_tags tag
        JOIN _generation_output_reference_targets target
          ON target.source_output_id = tag.asset_id
        WHERE tag.asset_type = 'output'
        ON CONFLICT (tag_id, asset_type, asset_id) DO NOTHING
        """
    )
    op.execute(
        """
        DELETE FROM library_asset_tags tag
        USING _generation_output_reference_targets target
        WHERE tag.asset_type = 'output'
          AND tag.asset_id = target.source_output_id
          AND target.source_output_id <> target.canonical_output_id
        """
    )

    op.execute(
        """
        DELETE FROM generation_outputs output
        USING _generation_output_duplicate_map map
        WHERE output.id = map.duplicate_id
        """
    )


def upgrade() -> None:
    """Add failure, claim, and output-materialization safeguards."""
    op.add_column(
        "generation_jobs",
        sa.Column("failure_code", sa.String(length=100), nullable=True),
    )
    op.add_column(
        "generation_jobs",
        sa.Column("public_error_message", sa.Text(), nullable=True),
    )
    op.add_column(
        "generation_jobs",
        sa.Column("finalization_claim_token", sa.String(length=36), nullable=True),
    )
    op.add_column(
        "generation_jobs",
        sa.Column("finalization_lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    # Existing rows originate before these fields.  Keep the repair explicit
    # if an interrupted manual deployment created an invalid pair somehow.
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (
                SELECT 1 FROM generation_jobs
                WHERE (finalization_claim_token IS NULL)
                    <> (finalization_lease_expires_at IS NULL)
            ) THEN
                RAISE EXCEPTION
                    '033 preflight found inconsistent finalization claim fields';
            END IF;
        END $$;
        """
    )
    op.create_check_constraint(
        "ck_generation_jobs_finalization_claim_pair",
        "generation_jobs",
        "(finalization_claim_token IS NULL) = (finalization_lease_expires_at IS NULL)",
    )
    op.create_index(
        "ix_generation_jobs_finalization_lease",
        "generation_jobs",
        ["finalization_lease_expires_at"],
        postgresql_where=sa.text("finalization_claim_token IS NOT NULL"),
    )

    op.create_table(
        "generation_materialization_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("generation_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("claim_token", sa.String(36), nullable=False),
        sa.Column("state", sa.String(24), nullable=False, server_default=sa.text("'planned'")),
        sa.Column(
            "planned_storage_keys",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column(
            "uploaded_storage_keys",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("reconciliation_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("claim_token"),
        sa.CheckConstraint(
            "state IN ('planned', 'uploading', 'committed', 'cleanup_pending', 'cleaned', 'ambiguous')",
            name="ck_generation_materialization_attempt_state",
        ),
    )
    op.create_index(
        "ix_generation_materialization_attempts_job_id",
        "generation_materialization_attempts",
        ["job_id"],
    )
    op.create_index(
        "ix_generation_materialization_attempts_product_id",
        "generation_materialization_attempts",
        ["product_id"],
    )
    op.create_index(
        "ix_generation_materialization_attempts_reconcile",
        "generation_materialization_attempts",
        ["product_id", "state", "updated_at"],
        postgresql_where=sa.text(
            "state IN ('planned', 'uploading', 'cleanup_pending', 'ambiguous')"
        ),
    )

    op.create_table(
        "storage_cleanup_records",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("reason", sa.String(80), nullable=False),
        sa.Column("state", sa.String(20), nullable=False, server_default=sa.text("'pending'")),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default=sa.text("0")),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("processed_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("storage_key"),
        sa.CheckConstraint(
            "state IN ('pending', 'deleted')", name="ck_storage_cleanup_records_state"
        ),
    )
    op.create_index(
        "ix_storage_cleanup_records_pending",
        "storage_cleanup_records",
        ["product_id", "created_at"],
        postgresql_where=sa.text("state = 'pending'"),
    )
    op.create_index(
        "ix_storage_cleanup_records_product_id",
        "storage_cleanup_records",
        ["product_id"],
    )

    # Audit incomplete legacy thumbnails before any cleanup.  They are not
    # canonicalizable from their bucket alone and deliberately remain intact.
    op.execute(
        """
        DO $$
        DECLARE incomplete_count bigint;
        BEGIN
            SELECT count(*) INTO incomplete_count
            FROM generation_outputs
            WHERE is_thumbnail = TRUE
              AND (parent_output_id IS NULL OR thumbnail_max_edge IS NULL);
            RAISE NOTICE
                '035 left % legacy thumbnails with incomplete canonical keys untouched',
                incomplete_count;
        END $$;
        """
    )

    op.execute(
        """
        CREATE TEMP TABLE _generation_output_duplicate_map (
            duplicate_id uuid PRIMARY KEY,
            canonical_id uuid NOT NULL
        ) ON COMMIT DROP
        """
    )
    _populate_duplicate_map(thumbnails=False)
    _preflight_and_remap_duplicates()
    op.execute("TRUNCATE _generation_output_duplicate_map")
    _populate_duplicate_map(thumbnails=True)
    _preflight_and_remap_duplicates()

    op.create_index(
        "uq_generation_outputs_full_job_index",
        "generation_outputs",
        ["job_id", "output_index"],
        unique=True,
        postgresql_where=sa.text("is_thumbnail = FALSE"),
    )
    op.create_index(
        "uq_generation_outputs_thumbnail_parent_bucket",
        "generation_outputs",
        ["parent_output_id", "thumbnail_max_edge"],
        unique=True,
        postgresql_where=sa.text(
            "is_thumbnail = TRUE AND parent_output_id IS NOT NULL "
            "AND thumbnail_max_edge IS NOT NULL"
        ),
    )

    # Seed the new class for already-linked admins.  Like the other rare
    # operational classes, this is ordinary preference state and can be
    # removed by the admin's next full-set preference update.
    op.execute(
        sa.text("""
            INSERT INTO admin_notification_preferences
                (id, user_id, product_id, notification_class, min_interval_seconds)
            SELECT gen_random_uuid(), atl.user_id, atl.product_id, :notification_class, 0
            FROM admin_telegram_links atl
            ON CONFLICT (user_id, product_id, notification_class) DO NOTHING
        """).bindparams(notification_class=_NOTIFICATION_CLASS)
    )


def downgrade() -> None:
    """Remove safeguards; already-committed outbox work remains auditable."""
    op.execute(
        sa.text("""
            DELETE FROM admin_notification_preferences
            WHERE notification_class = :notification_class
        """).bindparams(notification_class=_NOTIFICATION_CLASS)
    )
    op.drop_index("uq_generation_outputs_thumbnail_parent_bucket", table_name="generation_outputs")
    op.drop_index("uq_generation_outputs_full_job_index", table_name="generation_outputs")
    op.drop_index("ix_storage_cleanup_records_product_id", table_name="storage_cleanup_records")
    op.drop_index("ix_storage_cleanup_records_pending", table_name="storage_cleanup_records")
    op.drop_table("storage_cleanup_records")
    op.drop_index(
        "ix_generation_materialization_attempts_reconcile",
        table_name="generation_materialization_attempts",
    )
    op.drop_index(
        "ix_generation_materialization_attempts_job_id",
        table_name="generation_materialization_attempts",
    )
    op.drop_index(
        "ix_generation_materialization_attempts_product_id",
        table_name="generation_materialization_attempts",
    )
    op.drop_table("generation_materialization_attempts")
    op.drop_index("ix_generation_jobs_finalization_lease", table_name="generation_jobs")
    op.drop_constraint(
        "ck_generation_jobs_finalization_claim_pair",
        "generation_jobs",
        type_="check",
    )
    op.drop_column("generation_jobs", "finalization_lease_expires_at")
    op.drop_column("generation_jobs", "finalization_claim_token")
    op.drop_column("generation_jobs", "public_error_message")
    op.drop_column("generation_jobs", "failure_code")
