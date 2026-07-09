"""Regression tests for the onupdate=datetime.now(UTC) bug on User and GenerationModel.

Before the fix, updated_at was frozen at the process-startup timestamp because
datetime.now(UTC) was evaluated once at module-import time.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from src.db.models.generation_model import GenerationModel

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.db.models.user import User


async def test_user_updated_at_changes_on_update(
    db_session: AsyncSession,
    make_user: Any,
) -> None:
    """Regression for the onupdate=datetime.now(UTC) bug on User.

    Before the fix, updated_at was frozen at the process-startup timestamp
    because datetime.now(UTC) was evaluated once at module-import time.
    """
    user: User = await make_user()
    initial_updated_at = user.updated_at

    # Sleep briefly so CURRENT_TIMESTAMP on the UPDATE is strictly later.
    await asyncio.sleep(0.01)

    user.display_name = "changed"
    await db_session.flush()
    await db_session.refresh(user)

    assert user.updated_at > initial_updated_at, (
        f"updated_at should advance on UPDATE; "
        f"got {user.updated_at} <= initial {initial_updated_at}. "
        "Likely regression of the onupdate=datetime.now(UTC) bug."
    )


async def test_generation_model_updated_at_changes_on_update(
    db_session: AsyncSession,
) -> None:
    """Regression for the onupdate=datetime.now(UTC) bug on GenerationModel.

    Before the fix, updated_at was frozen at the process-startup timestamp
    because datetime.now(UTC) was evaluated once at module-import time.
    """
    model_key = f"test-model-{uuid4().hex[:8]}"
    gm = GenerationModel(
        model_key=model_key,
        provider="aisha",
        name="Test Model",
        is_enabled=True,
    )
    db_session.add(gm)
    await db_session.flush()
    await db_session.refresh(gm)
    initial_updated_at = gm.updated_at

    # Sleep briefly so CURRENT_TIMESTAMP on the UPDATE is strictly later.
    await asyncio.sleep(0.01)

    gm.is_enabled = False
    await db_session.flush()
    await db_session.refresh(gm)

    assert gm.updated_at > initial_updated_at, (
        f"updated_at should advance on UPDATE; "
        f"got {gm.updated_at} <= initial {initial_updated_at}. "
        "Likely regression of the onupdate=datetime.now(UTC) bug."
    )
