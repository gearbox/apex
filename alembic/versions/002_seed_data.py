"""Seed data — system user, pricing catalog, generation models.

All reference data required for application operation.
Schema is defined in 001_schema_baseline.py.

Revision ID: 002
Revises: 001
Create Date: 2026-03-15
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op
from src.api.schemas.grok import GROK_MODELS
from src.core.constants import SYSTEM_USER_ID
from src.core.enums import ModelType
from src.core.uid import new_id

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Static reference data
# ---------------------------------------------------------------------------

_INITIAL_PRICING: list[dict[str, object]] = [
    {"provider": "aisha", "generation_type": "t2i", "model": None, "token_cost": 10},
    {"provider": "aisha", "generation_type": "i2i", "model": None, "token_cost": 10},
    {"provider": "grok", "generation_type": "t2i", "model": "grok-imagine-image", "token_cost": 20},
    {"provider": "grok", "generation_type": "i2i", "model": "grok-imagine-image", "token_cost": 25},
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

_generation_models_t = sa.table(
    "generation_models",
    sa.column("id", postgresql.UUID(as_uuid=True)),
    sa.column("model_key", sa.String),
    sa.column("provider", sa.String),
    sa.column("name", sa.String),
    sa.column("description", sa.Text),
    sa.column("is_enabled", sa.Boolean),
)


def upgrade() -> None:
    """Insert all seed / reference data."""

    # -------------------------------------------------------------------------
    # 1. System user sentinel
    #    id is the fixed constant — NOT a generated UUIDv7.
    #    role='system', is_active=false prevent authentication.
    # -------------------------------------------------------------------------
    op.execute(
        sa.text("""
            INSERT INTO users (id, email, password_hash, role, is_active, subscription_tier, locale)
            VALUES (
                :id, 'system@internal', '', 'system', false, 'free', 'en'
            )
            ON CONFLICT DO NOTHING
        """).bindparams(id=SYSTEM_USER_ID)
    )

    # -------------------------------------------------------------------------
    # 2. Pricing catalog
    # -------------------------------------------------------------------------
    for rule in _INITIAL_PRICING:
        model_val = f"'{rule['model']}'" if rule["model"] else "NULL"
        op.execute(
            sa.text(f"""
                INSERT INTO pricing_catalog
                    (id, provider, generation_type, model, token_cost, is_active, created_by)
                VALUES (
                    :id,
                    '{rule["provider"]}',
                    '{rule["generation_type"]}',
                    {model_val},
                    {rule["token_cost"]},
                    true,
                    :created_by
                )
            """).bindparams(id=new_id(), created_by=SYSTEM_USER_ID)
        )

    # -------------------------------------------------------------------------
    # 3. Generation models registry
    # -------------------------------------------------------------------------
    seed_rows = [
        {
            "id": new_id(),
            "model_key": ModelType.AISHA.value,
            "provider": ModelType.AISHA.provider.value,
            "name": "Aisha",
            "description": "ComfyUI-based image and video generation via GPU workflows",
            "is_enabled": True,
        },
    ]
    seed_rows.extend(
        {
            "id": new_id(),
            "model_key": gm.model.value,
            "provider": gm.model.provider.value,
            "name": gm.name,
            "description": gm.description,
            "is_enabled": True,
        }
        for gm in GROK_MODELS
    )
    op.bulk_insert(_generation_models_t, seed_rows)


def downgrade() -> None:
    """Remove all seed data (reverse of upgrade)."""
    op.execute("DELETE FROM generation_models")
    op.execute("DELETE FROM pricing_catalog")
    op.execute(sa.text("DELETE FROM users WHERE id = :id").bindparams(id=SYSTEM_USER_ID))
