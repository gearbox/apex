"""Integration tests for GenerationModelRepository against a real PostgreSQL database."""

from __future__ import annotations

from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from src.db.models.generation_model import GenerationModel
from src.db.repositories.generation_model import GenerationModelRepository

# ---------------------------------------------------------------------------
# list_all
# ---------------------------------------------------------------------------


async def test_list_all_returns_seeded_rows(
    generation_model_repo: GenerationModelRepository,
) -> None:
    """list_all returns all seeded rows from migration 007."""
    rows = await generation_model_repo.list_all()
    keys = {r.model_key for r in rows}
    assert "aisha-image" in keys
    assert "aisha-video" in keys
    assert "grok-imagine-image" in keys
    assert "grok-2-image-1212" in keys
    assert "grok-imagine-video" in keys


async def test_list_all_ordered_by_provider_then_model_key(
    generation_model_repo: GenerationModelRepository,
) -> None:
    """list_all returns rows ordered by provider, model_key."""
    rows = await generation_model_repo.list_all()
    pairs = [(r.provider, r.model_key) for r in rows]
    assert pairs == sorted(pairs)


async def test_list_all_includes_disabled_rows(
    generation_model_repo: GenerationModelRepository,
    db_session: AsyncSession,
) -> None:
    """list_all returns disabled rows alongside enabled ones."""
    key = f"test-disabled-{uuid4().hex[:8]}"
    db_session.add(GenerationModel(model_key=key, provider="grok", name="Test", is_enabled=False))
    await db_session.flush()

    rows = await generation_model_repo.list_all()
    assert any(r.model_key == key for r in rows)


# ---------------------------------------------------------------------------
# list_enabled
# ---------------------------------------------------------------------------


async def test_list_enabled_returns_only_enabled_seeded_rows(
    generation_model_repo: GenerationModelRepository,
) -> None:
    """list_enabled returns the seeded rows which all start enabled."""
    rows = await generation_model_repo.list_enabled()
    assert all(r.is_enabled for r in rows)
    keys = {r.model_key for r in rows}
    assert "aisha-image" in keys
    assert "grok-imagine-image" in keys


async def test_list_enabled_excludes_disabled_row(
    generation_model_repo: GenerationModelRepository,
    db_session: AsyncSession,
) -> None:
    """list_enabled does not include rows with is_enabled=False."""
    key = f"test-off-{uuid4().hex[:8]}"
    db_session.add(GenerationModel(model_key=key, provider="grok", name="Off", is_enabled=False))
    await db_session.flush()

    rows = await generation_model_repo.list_enabled()
    assert all(r.model_key != key for r in rows)


async def test_list_enabled_includes_newly_enabled_row(
    generation_model_repo: GenerationModelRepository,
    db_session: AsyncSession,
) -> None:
    """list_enabled includes a row that was just inserted with is_enabled=True."""
    key = f"test-on-{uuid4().hex[:8]}"
    db_session.add(GenerationModel(model_key=key, provider="aisha", name="On", is_enabled=True))
    await db_session.flush()

    rows = await generation_model_repo.list_enabled()
    assert any(r.model_key == key for r in rows)


# ---------------------------------------------------------------------------
# get_by_key
# ---------------------------------------------------------------------------


async def test_get_by_key_returns_seeded_row(
    generation_model_repo: GenerationModelRepository,
) -> None:
    """get_by_key returns the correct seeded row."""
    row = await generation_model_repo.get_by_key("grok-imagine-image")
    assert row is not None
    assert row.model_key == "grok-imagine-image"
    assert row.provider == "grok"


async def test_get_by_key_returns_none_for_unknown_key(
    generation_model_repo: GenerationModelRepository,
) -> None:
    """get_by_key returns None for a key that does not exist."""
    row = await generation_model_repo.get_by_key("no-such-model")
    assert row is None


# ---------------------------------------------------------------------------
# set_enabled
# ---------------------------------------------------------------------------


async def test_set_enabled_false_persists(
    generation_model_repo: GenerationModelRepository,
    db_session: AsyncSession,
) -> None:
    """set_enabled(key, False) persists and the row is excluded by list_enabled."""
    key = f"test-toggle-{uuid4().hex[:8]}"
    db_session.add(GenerationModel(model_key=key, provider="grok", name="Toggle", is_enabled=True))
    await db_session.flush()

    updated = await generation_model_repo.set_enabled(key, False)
    assert updated is not None
    assert updated.is_enabled is False

    enabled_keys = {r.model_key for r in await generation_model_repo.list_enabled()}
    assert key not in enabled_keys


async def test_set_enabled_true_reappears_in_list_enabled(
    generation_model_repo: GenerationModelRepository,
    db_session: AsyncSession,
) -> None:
    """set_enabled(key, True) on a disabled row makes it reappear in list_enabled."""
    key = f"test-reenable-{uuid4().hex[:8]}"
    db_session.add(
        GenerationModel(model_key=key, provider="aisha", name="Reenable", is_enabled=False)
    )
    await db_session.flush()

    await generation_model_repo.set_enabled(key, True)

    enabled_keys = {r.model_key for r in await generation_model_repo.list_enabled()}
    assert key in enabled_keys


async def test_set_enabled_updates_updated_at(
    generation_model_repo: GenerationModelRepository,
    db_session: AsyncSession,
) -> None:
    """set_enabled updates the updated_at timestamp."""
    key = f"test-ts-{uuid4().hex[:8]}"
    db_session.add(GenerationModel(model_key=key, provider="grok", name="TS", is_enabled=True))
    await db_session.flush()

    row = await generation_model_repo.get_by_key(key)
    assert row is not None
    original_ts = row.updated_at

    updated = await generation_model_repo.set_enabled(key, False)
    assert updated is not None
    assert updated.updated_at >= original_ts


async def test_set_enabled_nonexistent_key_returns_none(
    generation_model_repo: GenerationModelRepository,
) -> None:
    """set_enabled returns None when the model_key does not exist."""
    result = await generation_model_repo.set_enabled("does-not-exist", True)
    assert result is None
