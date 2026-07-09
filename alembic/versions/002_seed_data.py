"""Seed data — system user, pricing catalog, generation models.

All reference data required for application operation.
Schema is defined in 001_schema_baseline.py.

This migration is fully self-contained: all seed values are inline
literals. It does NOT import from application code (schemas, enums)
to avoid coupling migration history to evolving source code.

Revision ID: 002
Revises: 001
Create Date: 2026-03-15
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# ---------------------------------------------------------------------------
# Fixed UUIDs — deterministic across all environments
# ---------------------------------------------------------------------------

SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"

# Generation model IDs (fixed for FK stability if ever needed)
_MODEL_ID_AISHA_IMAGE = "00000000-0000-0000-0001-000000000001"
_MODEL_ID_AISHA_VIDEO = "00000000-0000-0000-0001-000000000005"
_MODEL_ID_GROK_IMAGINE_IMAGE = "00000000-0000-0000-0001-000000000002"
_MODEL_ID_GROK_2_IMAGE = "00000000-0000-0000-0001-000000000003"
_MODEL_ID_GROK_IMAGINE_VIDEO = "00000000-0000-0000-0001-000000000004"

# Pricing rule IDs (fixed for audit trail reproducibility)
_PRICE_ID_AISHA_IMG_T2I = "00000000-0000-0000-0002-000000000001"
_PRICE_ID_AISHA_IMG_I2I = "00000000-0000-0000-0002-000000000002"
_PRICE_ID_AISHA_VID_T2V = "00000000-0000-0000-0002-000000000010"
_PRICE_ID_GROK_IMG_T2I = "00000000-0000-0000-0002-000000000003"
_PRICE_ID_GROK_IMG_I2I = "00000000-0000-0000-0002-000000000004"
_PRICE_ID_GROK_VID_T2V = "00000000-0000-0000-0002-000000000005"
_PRICE_ID_GROK_VID_I2V = "00000000-0000-0000-0002-000000000006"
_PRICE_ID_GROK_VID_V2V = "00000000-0000-0000-0002-000000000007"
_PRICE_ID_GROK_VID_FLF2V = "00000000-0000-0000-0002-000000000008"
_PRICE_ID_GROK_2IMG_T2I = "00000000-0000-0000-0002-000000000009"
_PRICE_ID_GPU_SESSION_START = "00000000-0000-0000-0002-000000000020"


# ---------------------------------------------------------------------------
# Seed data — all values are inline literals
# ---------------------------------------------------------------------------

_GENERATION_MODELS = [
    {
        "id": UUID(_MODEL_ID_AISHA_IMAGE),
        "model_key": "aisha-image",
        "provider": "aisha",
        "name": "Aisha Image",
        "description": "ComfyUI-based image generation via GPU workflows",
        "is_enabled": False,  # Disabled until image workflow is production-ready
    },
    {
        "id": UUID(_MODEL_ID_AISHA_VIDEO),
        "model_key": "aisha-video",
        "provider": "aisha",
        "name": "Aisha Video",
        "description": "ComfyUI-based video generation via GPU workflows",
        "is_enabled": False,  # Disabled until video workflow is production-ready
    },
    {
        "id": UUID(_MODEL_ID_GROK_IMAGINE_IMAGE),
        "model_key": "grok-imagine-image",
        "provider": "grok",
        "name": "Grok Imagine Image",
        "description": "xAI's flagship image generation model with editing support",
        "is_enabled": True,
    },
    {
        "id": UUID(_MODEL_ID_GROK_2_IMAGE),
        "model_key": "grok-2-image-1212",
        "provider": "grok",
        "name": "Grok 2 Image",
        "description": "Photorealistic image generation (legacy model)",
        "is_enabled": False,  # EOL after Grok Imagine Image release, kept for reference and potential fallback
    },
    {
        "id": UUID(_MODEL_ID_GROK_IMAGINE_VIDEO),
        "model_key": "grok-imagine-video",
        "provider": "grok",
        "name": "Grok Imagine Video",
        "description": "Video generation from text, images, or video editing (up to 15 seconds)",
        "is_enabled": True,
    },
]

_PRICING_RULES = [
    # Aisha Image
    {
        "id": _PRICE_ID_AISHA_IMG_T2I,
        "provider": "aisha",
        "generation_type": "t2i",
        "model": "aisha-image",
        "token_cost": 10,
    },
    {
        "id": _PRICE_ID_AISHA_IMG_I2I,
        "provider": "aisha",
        "generation_type": "i2i",
        "model": "aisha-image",
        "token_cost": 10,
    },
    # Aisha Video
    {
        "id": _PRICE_ID_AISHA_VID_T2V,
        "provider": "aisha",
        "generation_type": "t2v",
        "model": "aisha-video",
        "token_cost": 50,
    },
    # Grok Imagine Image
    {
        "id": _PRICE_ID_GROK_IMG_T2I,
        "provider": "grok",
        "generation_type": "t2i",
        "model": "grok-imagine-image",
        "token_cost": 20,
    },
    {
        "id": _PRICE_ID_GROK_IMG_I2I,
        "provider": "grok",
        "generation_type": "i2i",
        "model": "grok-imagine-image",
        "token_cost": 25,
    },
    # Grok 2 Image (legacy)
    {
        "id": _PRICE_ID_GROK_2IMG_T2I,
        "provider": "grok",
        "generation_type": "t2i",
        "model": "grok-2-image-1212",
        "token_cost": 15,
    },
    # GPU session base reservation
    {
        "id": _PRICE_ID_GPU_SESSION_START,
        "provider": "gpu_session",
        "generation_type": "session_start",
        "model": None,
        "token_cost": 500,
    },
    # Grok Imagine Video — all supported generation types
    {
        "id": _PRICE_ID_GROK_VID_T2V,
        "provider": "grok",
        "generation_type": "t2v",
        "model": "grok-imagine-video",
        "token_cost": 100,
    },
    {
        "id": _PRICE_ID_GROK_VID_I2V,
        "provider": "grok",
        "generation_type": "i2v",
        "model": "grok-imagine-video",
        "token_cost": 120,
    },
    {
        "id": _PRICE_ID_GROK_VID_V2V,
        "provider": "grok",
        "generation_type": "v2v",
        "model": "grok-imagine-video",
        "token_cost": 120,
    },
    {
        "id": _PRICE_ID_GROK_VID_FLF2V,
        "provider": "grok",
        "generation_type": "flf2v",
        "model": "grok-imagine-video",
        "token_cost": 130,
    },
]


# ---------------------------------------------------------------------------
# Table references for bulk_insert (must match 001 schema exactly)
# ---------------------------------------------------------------------------

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
    #    Fixed UUID — NOT a generated UUIDv7.
    #    role='system', is_active=false prevent authentication.
    # -------------------------------------------------------------------------
    op.execute(
        sa.text("""
            INSERT INTO users (id, email, password_hash, role, is_active, subscription_tier, locale, product_id)
            VALUES (
                :id, 'system@internal', '', 'system', false, 'free', 'en', 'vex'
            )
            ON CONFLICT DO NOTHING
        """).bindparams(id=UUID(SYSTEM_USER_ID))
    )

    # -------------------------------------------------------------------------
    # 2. Pricing catalog
    # -------------------------------------------------------------------------
    for rule in _PRICING_RULES:
        op.execute(
            sa.text("""
                INSERT INTO pricing_catalog
                    (id, provider, generation_type, model, token_cost, is_active, created_by)
                VALUES (
                    :id, :provider, :generation_type, :model, :token_cost, true, :created_by
                )
                ON CONFLICT DO NOTHING
            """).bindparams(
                id=UUID(str(rule["id"])),
                provider=rule["provider"],
                generation_type=rule["generation_type"],
                model=rule["model"],
                token_cost=rule["token_cost"],
                created_by=UUID(SYSTEM_USER_ID),
            )
        )

    # -------------------------------------------------------------------------
    # 3. Generation models registry
    # -------------------------------------------------------------------------
    op.bulk_insert(_generation_models_t, _GENERATION_MODELS)


def downgrade() -> None:
    """Remove all seed data (reverse of upgrade)."""
    op.execute("DELETE FROM generation_models")
    op.execute("DELETE FROM pricing_catalog")
    op.execute(sa.text("DELETE FROM users WHERE id = :id").bindparams(id=UUID(SYSTEM_USER_ID)))
