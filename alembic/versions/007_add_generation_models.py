"""Add generation_models table with per-model enable/disable flag.

Revision ID: 007
Revises: 006
Create Date: 2026-03-11 12:00:00.000000
"""


from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from src.api.schemas.grok import GROK_MODELS
from src.core.enums import ModelType

# revision identifiers
revision: str = "007"
down_revision: str | None = "006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Lightweight table clause used only for bulk_insert
_generation_models_t = sa.table(
    "generation_models",
    sa.column("model_key", sa.String),
    sa.column("provider", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("is_enabled", sa.Boolean),
)

# Seed rows derived from GROK_MODELS registry + AISHA ComfyUI model
_SEED_DATA: list[dict[str, object]] = [
    {
        "model_key": ModelType.AISHA.value,
        "provider": ModelType.AISHA.provider.value,
        "name": "Aisha",
        "description": "ComfyUI-based image and video generation via GPU workflows",
        "is_enabled": True,
    },
]
_SEED_DATA.extend(
    {
        "model_key": _grok_model.model.value,
        "provider": _grok_model.model.provider.value,
        "name": _grok_model.name,
        "description": _grok_model.description,
        "is_enabled": True,
    }
    for _grok_model in GROK_MODELS
)


def upgrade() -> None:
    """Create generation_models table and seed initial rows."""
    op.create_table(
        "generation_models",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            server_default=sa.text("gen_random_uuid()"),
            primary_key=True,
        ),
        sa.Column("model_key", sa.String(50), nullable=False),
        sa.Column("provider", sa.String(20), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column(
            "description",
            sa.Text,
            nullable=False,
            server_default=sa.text("''"),
        ),
        sa.Column(
            "is_enabled",
            sa.Boolean,
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
        sa.UniqueConstraint("model_key", name="uq_generation_models_model_key"),
    )
    op.create_index("ix_generation_models_provider", "generation_models", ["provider"])
    op.create_index("ix_generation_models_is_enabled", "generation_models", ["is_enabled"])

    op.bulk_insert(_generation_models_t, _SEED_DATA)


def downgrade() -> None:
    """Drop generation_models table."""
    op.drop_index("ix_generation_models_is_enabled", table_name="generation_models")
    op.drop_index("ix_generation_models_provider", table_name="generation_models")
    op.drop_table("generation_models")
