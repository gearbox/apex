"""Add billing tables

Revision ID: 004
Revises: 003
Create Date: 2026-02-23 12:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from src.core.constants import SYSTEM_USER_ID

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: str | None = "003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

INITIAL_PRICING = [
    # ComfyUI
    {"provider": "comfyui", "generation_type": "t2i", "model": None, "token_cost": 10},
    {"provider": "comfyui", "generation_type": "i2i", "model": None, "token_cost": 10},
    # Grok Image
    {
        "provider": "grok",
        "generation_type": "t2i",
        "model": "grok-imagine-image",
        "token_cost": 20,
    },
    {
        "provider": "grok",
        "generation_type": "i2i",
        "model": "grok-imagine-image",
        "token_cost": 25,
    },
    # Grok Video
    {
        "provider": "grok",
        "generation_type": "t2v",
        "model": "grok-imagine-video",
        "token_cost": 100,
    },
    {
        "provider": "grok",
        "generation_type": "i2v",
        "model": "grok-imagine-video",
        "token_cost": 120,
    },
]


def upgrade() -> None:
    """Upgrade database schema."""
    # 1. Create organizations table
    op.create_table(
        "organizations",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("slug", sa.String(100), unique=True, nullable=False),
        sa.Column(
            "owner_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
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
    )

    # 2. Create organization_members table
    op.create_table(
        "organization_members",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
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
        sa.Column(
            "joined_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.UniqueConstraint("organization_id", "user_id", name="uq_org_member"),
    )
    op.create_index(
        "ix_organization_members_user_id",
        "organization_members",
        ["user_id"],
    )

    # 3. Create token_accounts table
    op.create_table(
        "token_accounts",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("account_type", sa.String(20), nullable=False),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("users.id", ondelete="CASCADE"),
            unique=True,
            nullable=True,
        ),
        sa.Column(
            "organization_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("organizations.id", ondelete="CASCADE"),
            unique=True,
            nullable=True,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
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
        sa.CheckConstraint(
            """
            (account_type = 'personal' AND user_id IS NOT NULL AND organization_id IS NULL)
            OR
            (account_type = 'enterprise' AND organization_id IS NOT NULL AND user_id IS NULL)
            """,
            name="chk_account_owner",
        ),
    )

    # 4. Create payments table (before token_transactions — FK dependency)
    op.create_table(
        "payments",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column(
            "account_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("token_accounts.id"),
            nullable=False,
        ),
        sa.Column("payment_provider", sa.String(30), nullable=False),
        sa.Column("external_id", sa.String(255), unique=True, nullable=False),
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
        sa.CheckConstraint("amount_usd > 0", name="chk_amount_positive"),
        sa.CheckConstraint("tokens_granted > 0", name="chk_tokens_positive"),
    )
    op.create_index("ix_payments_account_id", "payments", ["account_id"])
    op.create_index("ix_payments_status_created", "payments", ["status", "created_at"])

    # 5. Create token_transactions table
    op.create_table(
        "token_transactions",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
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
            sa.ForeignKey("generation_jobs.id"),
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
        sa.CheckConstraint("amount != 0", name="chk_amount_nonzero"),
    )
    op.create_index(
        "ix_token_transactions_account_created",
        "token_transactions",
        ["account_id", "created_at"],
    )
    op.create_index(
        "ix_token_transactions_job_id",
        "token_transactions",
        ["job_id"],
    )
    op.create_index(
        "ix_token_transactions_payment_id",
        "token_transactions",
        ["payment_id"],
    )

    # 6. Create immutability trigger on token_transactions
    op.execute(
        """
        CREATE OR REPLACE FUNCTION prevent_token_transaction_mutation()
        RETURNS TRIGGER AS $$
        BEGIN
          RAISE EXCEPTION
            'token_transactions is append-only: UPDATE and DELETE are not permitted';
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        """
        CREATE TRIGGER enforce_token_transactions_immutable
          BEFORE UPDATE OR DELETE ON token_transactions
          FOR EACH ROW EXECUTE FUNCTION prevent_token_transaction_mutation();
        """
    )

    # 7. Create pricing_catalog table
    op.create_table(
        "pricing_catalog",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("generation_type", sa.String(20), nullable=False),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("token_cost", sa.Integer(), nullable=False),
        sa.Column(
            "is_active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
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
        sa.CheckConstraint("token_cost > 0", name="chk_token_cost_positive"),
        sa.CheckConstraint(
            "effective_until IS NULL OR effective_until > effective_from",
            name="chk_effective_range",
        ),
        sa.UniqueConstraint(
            "provider",
            "generation_type",
            "model",
            "effective_from",
            name="uq_pricing_rule",
        ),
    )

    # 8. Add billing columns to generation_jobs
    op.add_column(
        "generation_jobs",
        sa.Column("token_cost", sa.Integer(), nullable=True),
    )
    op.add_column(
        "generation_jobs",
        sa.Column(
            "debit_transaction_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("token_transactions.id"),
            nullable=True,
        ),
    )

    # 9. Data migration — create personal TokenAccount for all existing users
    op.execute(
        """
        INSERT INTO token_accounts (id, account_type, user_id, is_active, created_at, updated_at)
        SELECT gen_random_uuid(), 'personal', id, true, NOW(), NOW()
        FROM users
        """
    )

    # 10. Seed pricing_catalog with initial defaults
    for rule in INITIAL_PRICING:
        model_val = f"'{rule['model']}'" if rule["model"] else "NULL"
        op.execute(
            f"""
            INSERT INTO pricing_catalog
                (id, provider, generation_type, model, token_cost, is_active, created_by)
            VALUES (
                gen_random_uuid(),
                '{rule["provider"]}',
                '{rule["generation_type"]}',
                {model_val},
                {rule["token_cost"]},
                true,
                '{SYSTEM_USER_ID}'
            )
            """
        )


def downgrade() -> None:
    """Downgrade database schema."""
    # Remove billing columns from generation_jobs
    op.drop_column("generation_jobs", "debit_transaction_id")
    op.drop_column("generation_jobs", "token_cost")

    # Drop pricing_catalog
    op.drop_table("pricing_catalog")

    # Drop immutability trigger and function
    op.execute("DROP TRIGGER IF EXISTS enforce_token_transactions_immutable ON token_transactions")
    op.execute("DROP FUNCTION IF EXISTS prevent_token_transaction_mutation()")

    # Drop token_transactions
    op.drop_index("ix_token_transactions_payment_id", table_name="token_transactions")
    op.drop_index("ix_token_transactions_job_id", table_name="token_transactions")
    op.drop_index("ix_token_transactions_account_created", table_name="token_transactions")
    op.drop_table("token_transactions")

    # Drop payments
    op.drop_index("ix_payments_status_created", table_name="payments")
    op.drop_index("ix_payments_account_id", table_name="payments")
    op.drop_table("payments")

    # Drop token_accounts
    op.drop_table("token_accounts")

    # Drop organization_members
    op.drop_index("ix_organization_members_user_id", table_name="organization_members")
    op.drop_table("organization_members")

    # Drop organizations
    op.drop_table("organizations")
