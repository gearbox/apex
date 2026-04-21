"""Baseline schema — squash of migrations 001-008.

All tables, indexes, constraints, triggers, and functions.
No seed data — see 002_seed_data.py.

Revision ID: 001
Revises:
Create Date: 2026-03-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

revision: str = "001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create full schema from scratch."""

    # -------------------------------------------------------------------------
    # users
    # -------------------------------------------------------------------------
    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("display_name", sa.String(100), nullable=True),
        sa.Column("subscription_tier", sa.String(20), nullable=False, server_default="free"),
        sa.Column("role", sa.String(20), nullable=False, server_default="user"),
        sa.Column("preferred_billing_account", sa.String(20), nullable=True),
        sa.Column("locale", sa.String(10), nullable=False, server_default="en"),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("age_verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
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
        sa.PrimaryKeyConstraint("id"),
    )
    # Partial unique index: only active users must have unique (email, product_id)
    op.execute(
        "CREATE UNIQUE INDEX ix_users_email_product"
        " ON users(email, product_id)"
        " WHERE is_active = TRUE"
    )
    op.create_index("ix_users_email_active", "users", ["email", "is_active"])
    op.create_index("ix_users_product", "users", ["product_id"])
    op.create_index(op.f("ix_users_is_active"), "users", ["is_active"])

    # -------------------------------------------------------------------------
    # idempotency_keys
    # -------------------------------------------------------------------------
    op.create_table(
        "idempotency_keys",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("idempotency_key", sa.String(64), nullable=False),
        sa.Column("operation", sa.String(30), nullable=False),
        sa.Column(
            "status",
            sa.String(20),
            nullable=False,
            server_default=sa.text("'processing'"),
        ),
        sa.Column("resource_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("response_status_code", sa.SmallInteger(), nullable=True),
        sa.Column("response_body", postgresql.JSONB(), nullable=True),
        sa.Column("request_hash", sa.String(64), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "product_id",
            "idempotency_key",
            name="uq_idempotency_user_product_key",
        ),
    )
    op.create_index(
        "ix_idempotency_keys_expires_at",
        "idempotency_keys",
        ["expires_at"],
    )
    op.create_index(
        "ix_idempotency_keys_product_id",
        "idempotency_keys",
        ["product_id"],
    )

    # -------------------------------------------------------------------------
    # refresh_tokens
    # -------------------------------------------------------------------------
    op.create_table(
        "refresh_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(255), nullable=False),
        sa.Column("family_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("user_agent", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column("is_revoked", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_refresh_tokens_cleanup", "refresh_tokens", ["expires_at", "is_revoked"])
    op.create_index(op.f("ix_refresh_tokens_expires_at"), "refresh_tokens", ["expires_at"])
    op.create_index(op.f("ix_refresh_tokens_family_id"), "refresh_tokens", ["family_id"])
    op.create_index(
        op.f("ix_refresh_tokens_token_hash"), "refresh_tokens", ["token_hash"], unique=True
    )
    op.create_index(op.f("ix_refresh_tokens_user_id"), "refresh_tokens", ["user_id"])
    op.create_index(
        "ix_refresh_tokens_user_valid",
        "refresh_tokens",
        ["user_id", "is_revoked", "expires_at"],
    )
    op.create_index(op.f("ix_refresh_tokens_product_id"), "refresh_tokens", ["product_id"])

    # -------------------------------------------------------------------------
    # email_verification_tokens
    # -------------------------------------------------------------------------
    op.create_table(
        "email_verification_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_email_verification_tokens_user_id"), "email_verification_tokens", ["user_id"]
    )
    op.create_index(
        op.f("ix_email_verification_tokens_expires_at"),
        "email_verification_tokens",
        ["expires_at"],
    )
    op.create_index(
        op.f("ix_email_verification_tokens_token_hash"),
        "email_verification_tokens",
        ["token_hash"],
        unique=True,
    )
    op.create_index(
        "ix_email_verification_tokens_cleanup", "email_verification_tokens", ["expires_at"]
    )
    op.create_index(
        "ix_email_verification_tokens_user_active",
        "email_verification_tokens",
        ["user_id", "used_at"],
    )

    # -------------------------------------------------------------------------
    # password_reset_tokens
    # -------------------------------------------------------------------------
    op.create_table(
        "password_reset_tokens",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("token_hash", sa.String(64), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("used_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_password_reset_tokens_user_id"), "password_reset_tokens", ["user_id"])
    op.create_index(
        op.f("ix_password_reset_tokens_expires_at"), "password_reset_tokens", ["expires_at"]
    )
    op.create_index(
        op.f("ix_password_reset_tokens_token_hash"),
        "password_reset_tokens",
        ["token_hash"],
        unique=True,
    )
    op.create_index("ix_password_reset_tokens_cleanup", "password_reset_tokens", ["expires_at"])
    op.create_index(
        "ix_password_reset_tokens_user_active",
        "password_reset_tokens",
        ["user_id", "used_at"],
    )

    # -------------------------------------------------------------------------
    # organizations
    # -------------------------------------------------------------------------
    op.create_table(
        "organizations",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), nullable=False),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("product_id", sa.String(32), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug"),
    )

    # -------------------------------------------------------------------------
    # organization_members
    # -------------------------------------------------------------------------
    op.create_table(
        "organization_members",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("organization_id", "user_id", name="uq_org_member"),
    )
    op.create_index(op.f("ix_organization_members_user_id"), "organization_members", ["user_id"])
    op.create_index(
        op.f("ix_organization_members_product_id"), "organization_members", ["product_id"]
    )
    op.create_index(op.f("ix_organizations_product_id"), "organizations", ["product_id"])

    # -------------------------------------------------------------------------
    # token_accounts
    # -------------------------------------------------------------------------
    op.create_table(
        "token_accounts",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("account_type", sa.String(20), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            nullable=True,
        ),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("product_id", sa.String(32), nullable=False),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", name="uq_token_accounts_user_id"),
        sa.UniqueConstraint("organization_id", name="uq_token_accounts_organization_id"),
        sa.CheckConstraint(
            """
            (account_type = 'personal' AND user_id IS NOT NULL AND organization_id IS NULL)
            OR
            (account_type = 'enterprise' AND organization_id IS NOT NULL AND user_id IS NULL)
            """,
            name="chk_account_owner",
        ),
    )

    op.create_index(op.f("ix_token_accounts_product_id"), "token_accounts", ["product_id"])

    # -------------------------------------------------------------------------
    # payments
    # (created before token_transactions because token_transactions.payment_id
    #  references payments.id)
    # -------------------------------------------------------------------------
    op.create_table(
        "payments",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("token_accounts.id"),
            nullable=False,
        ),
        sa.Column("payment_provider", sa.String(30), nullable=False),
        sa.Column("external_id", sa.String(255), nullable=False),
        sa.Column("status", sa.String(30), nullable=False),
        sa.Column("amount_usd", sa.Numeric(10, 2), nullable=False),
        sa.Column("tokens_granted", sa.Integer(), nullable=False),
        sa.Column(
            "currency",
            sa.String(10),
            nullable=False,
            server_default=sa.text("'USD'"),
        ),
        sa.Column(
            "provider_metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("external_id"),
        sa.CheckConstraint("amount_usd > 0", name="chk_amount_positive"),
        sa.CheckConstraint("tokens_granted > 0", name="chk_tokens_positive"),
    )
    op.create_index(op.f("ix_payments_account_id"), "payments", ["account_id"])
    op.create_index("ix_payments_status_created", "payments", ["status", "created_at"])
    op.create_index(op.f("ix_payments_product_id"), "payments", ["product_id"])

    # -------------------------------------------------------------------------
    # token_transactions  (append-only ledger)
    # job_id FK to generation_jobs is added after generation_jobs is created.
    # -------------------------------------------------------------------------
    op.create_table(
        "token_transactions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("token_accounts.id"),
            nullable=False,
        ),
        sa.Column("transaction_type", sa.String(30), nullable=False),
        sa.Column("amount", sa.BigInteger(), nullable=False),
        sa.Column("balance_after", sa.BigInteger(), nullable=False),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
        sa.Column(
            "payment_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("payments.id"),
            nullable=True,
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.CheckConstraint("amount != 0", name="chk_amount_nonzero"),
    )
    op.create_index(
        "ix_token_transactions_account_created",
        "token_transactions",
        ["account_id", "created_at"],
    )
    op.create_index("ix_token_transactions_job_id", "token_transactions", ["job_id"])
    op.create_index("ix_token_transactions_payment_id", "token_transactions", ["payment_id"])
    op.create_index(op.f("ix_token_transactions_product_id"), "token_transactions", ["product_id"])

    # Immutability trigger — prevent UPDATE/DELETE on token_transactions
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_token_transaction_mutation()
        RETURNS trigger LANGUAGE plpgsql AS $$
        BEGIN
            RAISE EXCEPTION 'token_transactions rows are immutable';
        END;
        $$
    """)
    op.execute("""
        CREATE TRIGGER enforce_token_transactions_immutable
        BEFORE UPDATE OR DELETE ON token_transactions
        FOR EACH ROW EXECUTE FUNCTION prevent_token_transaction_mutation()
    """)

    # -------------------------------------------------------------------------
    # generation_jobs
    # debit_transaction_id FK to token_transactions is added after this table.
    # -------------------------------------------------------------------------
    op.create_table(
        "generation_jobs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("worker_id", sa.String(100), nullable=True),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("is_deleted", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("generation_type", sa.String(20), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False, server_default=sa.text("'aisha'")),
        sa.Column("model", sa.String(50), nullable=True),
        sa.Column("external_request_id", sa.String(64), nullable=True),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("enhanced_prompt", sa.Text(), nullable=True),
        sa.Column("negative_prompt", sa.Text(), nullable=True),
        sa.Column("aspect_ratio", sa.String(10), nullable=True),
        sa.Column("theme_detected", sa.String(100), nullable=True),
        sa.Column("theme_confidence", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("is_nsfw", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column(
            "is_minor_suspected", sa.Boolean(), nullable=False, server_default=sa.text("false")
        ),
        sa.Column("token_cost", sa.Integer(), nullable=True),
        sa.Column("debit_transaction_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("source_job_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("source_output_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("input_image_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_generation_jobs_generation_type"), "generation_jobs", ["generation_type"]
    )
    op.create_index(op.f("ix_generation_jobs_model"), "generation_jobs", ["model"])
    op.create_index(op.f("ix_generation_jobs_provider"), "generation_jobs", ["provider"])
    op.create_index("ix_generation_jobs_provider_status", "generation_jobs", ["provider", "status"])
    op.create_index(op.f("ix_generation_jobs_status"), "generation_jobs", ["status"])
    op.create_index(
        op.f("ix_generation_jobs_theme_detected"), "generation_jobs", ["theme_detected"]
    )
    op.create_index(op.f("ix_generation_jobs_user_id"), "generation_jobs", ["user_id"])
    op.create_index("ix_generation_jobs_user_created", "generation_jobs", ["user_id", "created_at"])
    op.create_index("ix_generation_jobs_user_status", "generation_jobs", ["user_id", "status"])
    op.create_index(op.f("ix_generation_jobs_product_id"), "generation_jobs", ["product_id"])
    op.create_index(
        "ix_generation_jobs_deleted",
        "generation_jobs",
        ["user_id", "is_deleted"],
        postgresql_where="is_deleted = TRUE",
    )

    # Cross-reference FK: generation_jobs.debit_transaction_id → token_transactions.id
    # (only this direction — token_transactions.job_id has NO FK because the ledger
    # also links non-job resources like GPU sessions; see src/db/models/billing.py)
    op.create_foreign_key(
        None,
        "generation_jobs",
        "token_transactions",
        ["debit_transaction_id"],
        ["id"],
    )

    # -------------------------------------------------------------------------
    # user_images
    # -------------------------------------------------------------------------
    op.create_table(
        "user_images",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("original_filename", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("format", sa.String(10), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index(op.f("ix_user_images_user_id"), "user_images", ["user_id"])
    op.create_index(op.f("ix_user_images_expires_at"), "user_images", ["expires_at"])
    op.create_index("ix_user_images_user_created", "user_images", ["user_id", "created_at"])
    op.create_index("ix_user_images_cleanup", "user_images", ["expires_at"])
    op.create_index(op.f("ix_user_images_product_id"), "user_images", ["product_id"])

    # -------------------------------------------------------------------------
    # generation_outputs
    # -------------------------------------------------------------------------
    op.create_table(
        "generation_outputs",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "job_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("generation_jobs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "input_image_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("user_images.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("storage_key", sa.String(512), nullable=False),
        sa.Column("content_type", sa.String(100), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("format", sa.String(10), nullable=False),
        sa.Column("output_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_thumbnail", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("storage_key"),
    )
    op.create_index(op.f("ix_generation_outputs_user_id"), "generation_outputs", ["user_id"])
    op.create_index(op.f("ix_generation_outputs_expires_at"), "generation_outputs", ["expires_at"])
    op.create_index("ix_generation_outputs_job", "generation_outputs", ["job_id"])
    op.create_index(
        "ix_generation_outputs_user_created", "generation_outputs", ["user_id", "created_at"]
    )
    op.create_index("ix_generation_outputs_cleanup", "generation_outputs", ["expires_at"])
    op.create_index(
        "ix_generation_outputs_thumbnail", "generation_outputs", ["job_id", "is_thumbnail"]
    )
    op.create_index(op.f("ix_generation_outputs_product_id"), "generation_outputs", ["product_id"])

    # Lineage FKs on generation_jobs (added here because user_images and
    # generation_outputs must exist first)
    op.create_foreign_key(
        "fk_generation_jobs_source_job",
        "generation_jobs",
        "generation_jobs",
        ["source_job_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_generation_jobs_source_output",
        "generation_jobs",
        "generation_outputs",
        ["source_output_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_generation_jobs_input_image",
        "generation_jobs",
        "user_images",
        ["input_image_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_generation_jobs_source_job_id",
        "generation_jobs",
        ["source_job_id"],
    )
    op.create_index(
        "ix_generation_jobs_gallery",
        "generation_jobs",
        ["user_id", "product_id", "status", "created_at"],
    )

    # -------------------------------------------------------------------------
    # pricing_catalog
    # -------------------------------------------------------------------------
    op.create_table(
        "pricing_catalog",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("generation_type", sa.String(20), nullable=False),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("token_cost", sa.Integer(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column(
            "effective_from",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column("effective_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "provider", "generation_type", "model", "effective_from", name="uq_pricing_rule"
        ),
        sa.CheckConstraint("token_cost > 0", name="chk_token_cost_positive"),
        sa.CheckConstraint(
            "effective_until IS NULL OR effective_until > effective_from",
            name="chk_effective_range",
        ),
    )

    # -------------------------------------------------------------------------
    # generation_models
    # -------------------------------------------------------------------------
    op.create_table(
        "generation_models",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("model_key", sa.String(50), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=sa.text("''")),
        sa.Column("is_enabled", sa.Boolean(), nullable=False, server_default=sa.text("true")),
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
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("model_key"),
    )
    op.create_index("ix_generation_models_provider", "generation_models", ["provider"])
    op.create_index("ix_generation_models_is_enabled", "generation_models", ["is_enabled"])

    # -------------------------------------------------------------------------
    # admin_permissions
    # -------------------------------------------------------------------------
    op.create_table(
        "admin_permissions",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("permission", sa.String(50), nullable=False),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column(
            "granted_by",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "granted_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id", "permission", "product_id", name="uq_admin_perm_user_perm_product"
        ),
    )
    op.create_index("ix_admin_perm_user_product", "admin_permissions", ["user_id", "product_id"])

    # -------------------------------------------------------------------------
    # admin_audit_log
    # -------------------------------------------------------------------------
    op.create_table(
        "admin_audit_log",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column(
            "actor_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "target_user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("product_id", sa.String(32), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("detail", sa.Text(), nullable=False),
        sa.Column("source", sa.String(10), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_audit_target_product", "admin_audit_log", ["target_user_id", "product_id"])
    op.create_index("ix_audit_created", "admin_audit_log", ["created_at"])


def downgrade() -> None:
    """Drop all tables in reverse dependency order."""
    op.drop_constraint(
        "generation_jobs_debit_transaction_id_fkey", "generation_jobs", type_="foreignkey"
    )
    op.drop_constraint("fk_generation_jobs_source_output", "generation_jobs", type_="foreignkey")
    op.drop_constraint("fk_generation_jobs_input_image", "generation_jobs", type_="foreignkey")
    op.drop_constraint("fk_generation_jobs_source_job", "generation_jobs", type_="foreignkey")

    op.drop_index("ix_audit_created", table_name="admin_audit_log")
    op.drop_index("ix_audit_target_product", table_name="admin_audit_log")
    op.drop_table("admin_audit_log")
    op.drop_index("ix_admin_perm_user_product", table_name="admin_permissions")
    op.drop_table("admin_permissions")
    op.drop_table("generation_models")
    op.drop_table("pricing_catalog")
    op.drop_table("generation_outputs")
    op.drop_table("user_images")
    op.drop_table("generation_jobs")
    op.execute("DROP TRIGGER IF EXISTS enforce_token_transactions_immutable ON token_transactions")
    op.execute("DROP FUNCTION IF EXISTS prevent_token_transaction_mutation()")
    op.drop_table("token_transactions")
    op.drop_table("payments")
    op.drop_table("token_accounts")
    op.drop_table("organization_members")
    op.drop_table("organizations")
    op.drop_table("password_reset_tokens")
    op.drop_table("email_verification_tokens")
    op.drop_table("refresh_tokens")
    op.drop_table("idempotency_keys")
    op.drop_table("users")
