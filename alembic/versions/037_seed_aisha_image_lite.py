"""Seed generation_models and pricing_catalog rows for aisha-image-lite.

zit.cyberrealistic — a second, lighter Aisha image model (t2i-only, no
negative_prompt). Priced below aisha-image (10 tokens flat) to reflect the
lighter footprint (11.4GB weights / 60GB min_disk_gb vs aisha-image's
28.4GB / 80GB), while keeping a comfortable margin so under-pricing never
becomes the failure mode.

Revision ID: 037
Revises: 036
Create Date: 2026-08-30 00:00:00.000000
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import sqlalchemy as sa

from alembic import op

if TYPE_CHECKING:
    from collections.abc import Sequence

revision: str = "037"
down_revision: str | Sequence[str] | None = "036"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

SYSTEM_USER_ID = "00000000-0000-0000-0000-000000000001"

_MODEL_ID_AISHA_IMAGE_LITE = "00000000-0000-0000-0001-000000000006"
_PRICE_ID_AISHA_IMG_LITE_T2I = "00000000-0000-0000-0002-000000000021"

_GENERATION_MODELS = [
    {
        "id": UUID(_MODEL_ID_AISHA_IMAGE_LITE),
        "model_key": "aisha-image-lite",
        "provider": "aisha",
        "name": "Aisha Image Lite",
        "description": "ComfyUI-based image generation (zit.cyberrealistic, t2i-only)",
        "is_enabled": False,  # Disabled until the bundle is indexed and verified
    },
]

_PRICING_RULES = [
    {
        "id": _PRICE_ID_AISHA_IMG_LITE_T2I,
        "provider": "aisha",
        "generation_type": "t2i",
        "model": "aisha-image-lite",
        "token_cost": 1,
    },
]


def upgrade() -> None:
    """Insert the aisha-image-lite generation model and its t2i pricing rule."""
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

    for model in _GENERATION_MODELS:
        op.execute(
            sa.text("""
                INSERT INTO generation_models
                    (id, model_key, provider, name, description, is_enabled)
                VALUES (
                    :id, :model_key, :provider, :name, :description, :is_enabled
                )
                ON CONFLICT DO NOTHING
            """).bindparams(
                id=UUID(str(model["id"])),
                model_key=model["model_key"],
                provider=model["provider"],
                name=model["name"],
                description=model["description"],
                is_enabled=model["is_enabled"],
            )
        )


def downgrade() -> None:
    """Remove only the rows this migration inserted."""
    op.execute(sa.text("DELETE FROM generation_models WHERE model_key = 'aisha-image-lite'"))
    op.execute(sa.text("DELETE FROM pricing_catalog WHERE model = 'aisha-image-lite'"))
