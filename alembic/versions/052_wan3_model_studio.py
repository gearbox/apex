"""Per-second pricing, video-resolution rules and the wan3.0-video model row.

Revision ID: 052
Revises: 051
Create Date: 2026-10-09 00:00:00.000000

``pricing_catalog`` gains ``per_second_token_cost`` (``NOT NULL DEFAULT 0``) and an
optional ``video_resolution`` match (``NULL`` = any resolution). A rule must still
cost something, so ``chk_token_cost_positive`` is replaced by
``chk_pricing_costs_valid`` (``token_cost > 0 OR per_second_token_cost > 0``).
``uq_pricing_rule`` now includes ``video_resolution`` and keeps its NULL semantics
(NULLs are distinct), exactly as before.

The ``wan3.0-video`` model row is seeded **disabled** and **no prices are seeded**:
an admin creates one rule per resolution from the Model Studio console prices and
then enables the model.
"""

from collections.abc import Sequence
from uuid import UUID

import sqlalchemy as sa

from alembic import op

revision: str = "052"
down_revision: str | Sequence[str] | None = "051"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_MODEL_ID_WAN_3_VIDEO = "00000000-0000-0000-0001-000000000007"

_OLD_UNIQUE_COLUMNS = ["provider", "generation_type", "model", "effective_from"]
_NEW_UNIQUE_COLUMNS = ["provider", "generation_type", "model", "video_resolution", "effective_from"]


def upgrade() -> None:
    """Upgrade database schema."""
    op.add_column(
        "pricing_catalog",
        sa.Column(
            "per_second_token_cost",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
    )
    op.add_column(
        "pricing_catalog",
        sa.Column("video_resolution", sa.String(length=10), nullable=True),
    )
    op.drop_constraint("chk_token_cost_positive", "pricing_catalog", type_="check")
    op.create_check_constraint(
        "chk_pricing_costs_valid",
        "pricing_catalog",
        "token_cost >= 0 AND per_second_token_cost >= 0 AND input_token_cost >= 0 "
        "AND (token_cost > 0 OR per_second_token_cost > 0)",
    )
    op.create_check_constraint(
        "chk_pricing_video_resolution",
        "pricing_catalog",
        "video_resolution IS NULL OR video_resolution IN ('480p', '720p', '1080p')",
    )
    op.drop_constraint("uq_pricing_rule", "pricing_catalog", type_="unique")
    op.create_unique_constraint("uq_pricing_rule", "pricing_catalog", _NEW_UNIQUE_COLUMNS)

    op.execute(
        sa.text("""
            INSERT INTO generation_models
                (id, model_key, provider, name, description, is_enabled)
            VALUES (
                :id, :model_key, :provider, :name, :description, :is_enabled
            )
            ON CONFLICT DO NOTHING
        """).bindparams(
            id=UUID(_MODEL_ID_WAN_3_VIDEO),
            model_key="wan3.0-video",
            provider="alibaba",
            name="Wan 3.0",
            description="Alibaba Model Studio video generation (t2v, i2v, flf2v, v2v, r2v)",
            is_enabled=False,  # Enabled by an admin once pricing rules exist
        )
    )


def downgrade() -> None:
    """Downgrade database schema, refusing to drop data the old schema cannot hold."""
    bind = op.get_bind()
    in_use = bind.execute(
        sa.text(
            "SELECT count(*) FROM pricing_catalog "
            "WHERE per_second_token_cost <> 0 OR video_resolution IS NOT NULL"
        )
    ).scalar_one()
    if in_use:
        raise RuntimeError(
            f"Cannot downgrade 052: {in_use} pricing_catalog row(s) use per_second_token_cost "
            "or video_resolution. Deactivate and delete them first."
        )

    op.execute(sa.text("DELETE FROM generation_models WHERE model_key = 'wan3.0-video'"))

    op.drop_constraint("uq_pricing_rule", "pricing_catalog", type_="unique")
    op.create_unique_constraint("uq_pricing_rule", "pricing_catalog", _OLD_UNIQUE_COLUMNS)
    op.drop_constraint("chk_pricing_video_resolution", "pricing_catalog", type_="check")
    op.drop_constraint("chk_pricing_costs_valid", "pricing_catalog", type_="check")
    op.create_check_constraint("chk_token_cost_positive", "pricing_catalog", "token_cost > 0")
    op.drop_column("pricing_catalog", "video_resolution")
    op.drop_column("pricing_catalog", "per_second_token_cost")
