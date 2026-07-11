"""Unit tests for the broken-node cooldown store and offer filter."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.api.services.gpu_session.node_cooldown import (
    NullNodeCooldownStore,
    RedisNodeCooldownStore,
    apply_cooldown_filter,
)
from src.api.services.vastai.schemas import VastAIOffer

pytestmark = pytest.mark.unit


def _make_settings(
    *,
    base_minutes: int = 30,
    max_minutes: int = 360,
    window_hours: int = 24,
) -> MagicMock:
    settings = MagicMock()
    settings.node_cooldown_base_minutes = base_minutes
    settings.node_cooldown_max_minutes = max_minutes
    settings.node_cooldown_failure_window_hours = window_hours
    return settings


def _make_redis_with_pipeline(execute_return: list[object]) -> MagicMock:
    """Redis mock whose .pipeline().execute() returns the given values."""
    redis = MagicMock()
    pipe = MagicMock()
    pipe.incr = MagicMock()
    pipe.expire = MagicMock()
    pipe.execute = AsyncMock(return_value=execute_return)
    redis.pipeline = MagicMock(return_value=pipe)
    redis.set = AsyncMock()
    redis.mget = AsyncMock()
    return redis


def _make_offer(*, offer_id: int, machine_id: int | None, dph_total: float = 0.5) -> VastAIOffer:
    return VastAIOffer(id=offer_id, gpu_name="RTX_4090", dph_total=dph_total, machine_id=machine_id)


class TestRedisNodeCooldownStore:
    async def test_record_failure_sets_cooldown_with_base_ttl(self) -> None:
        redis = _make_redis_with_pipeline([1, True])
        store = RedisNodeCooldownStore(redis, _make_settings(base_minutes=30, max_minutes=360))

        await store.record_failure(123, reason="provisioning_timeout")

        redis.set.assert_awaited_once_with(
            "vastai:node_cooldown:123", "provisioning_timeout", ex=30 * 60
        )

    async def test_record_failure_escalates_ttl(self) -> None:
        redis = _make_redis_with_pipeline([3, True])
        store = RedisNodeCooldownStore(redis, _make_settings(base_minutes=30, max_minutes=3600))

        await store.record_failure(123, reason="provisioning_stalled")

        expected_ttl = 30 * 60 * 4  # base * 2^(3-1)
        redis.set.assert_awaited_once_with(
            "vastai:node_cooldown:123", "provisioning_stalled", ex=expected_ttl
        )

    async def test_record_failure_ttl_capped_at_max(self) -> None:
        redis = _make_redis_with_pipeline([10, True])
        store = RedisNodeCooldownStore(redis, _make_settings(base_minutes=30, max_minutes=360))

        await store.record_failure(123, reason="node_reported_failure")

        redis.set.assert_awaited_once_with(
            "vastai:node_cooldown:123", "node_reported_failure", ex=360 * 60
        )

    async def test_record_failure_redis_error_is_swallowed(self) -> None:
        redis = MagicMock()
        pipe = MagicMock()
        pipe.incr = MagicMock()
        pipe.expire = MagicMock()
        pipe.execute = AsyncMock(side_effect=RuntimeError("redis down"))
        redis.pipeline = MagicMock(return_value=pipe)
        store = RedisNodeCooldownStore(redis, _make_settings())

        await store.record_failure(123, reason="test")  # must not raise

    async def test_cooled_machine_ids_returns_only_set_keys(self) -> None:
        redis = _make_redis_with_pipeline([])
        redis.mget.return_value = ["provisioning_timeout", None, "node_reported_failure"]
        store = RedisNodeCooldownStore(redis, _make_settings())

        result = await store.cooled_machine_ids([1, 2, 3])

        assert result == {1, 3}

    async def test_cooled_machine_ids_empty_input_short_circuits(self) -> None:
        redis = _make_redis_with_pipeline([])
        store = RedisNodeCooldownStore(redis, _make_settings())

        result = await store.cooled_machine_ids([])

        assert result == set()
        redis.mget.assert_not_called()

    async def test_cooled_machine_ids_redis_error_fails_open(self) -> None:
        redis = _make_redis_with_pipeline([])
        redis.mget.side_effect = RuntimeError("redis down")
        store = RedisNodeCooldownStore(redis, _make_settings())

        result = await store.cooled_machine_ids([1, 2])

        assert result == set()


class TestNullNodeCooldownStore:
    async def test_noops(self) -> None:
        store = NullNodeCooldownStore()

        await store.record_failure(123, reason="whatever")  # must not raise
        result = await store.cooled_machine_ids([1, 2, 3])

        assert result == set()


class TestApplyCooldownFilter:
    async def test_filters_cooled_offers_preserving_order(self) -> None:
        offers = [
            _make_offer(offer_id=1, machine_id=100),
            _make_offer(offer_id=2, machine_id=200),
            _make_offer(offer_id=3, machine_id=300),
        ]
        store = AsyncMock()
        store.cooled_machine_ids.return_value = {200}

        result = await apply_cooldown_filter(offers, store)

        assert [o.id for o in result] == [1, 3]

    async def test_offers_without_machine_id_pass_through(self) -> None:
        offers = [
            _make_offer(offer_id=1, machine_id=None),
            _make_offer(offer_id=2, machine_id=200),
        ]
        store = AsyncMock()
        store.cooled_machine_ids.return_value = {200}

        result = await apply_cooldown_filter(offers, store)

        assert [o.id for o in result] == [1]

    async def test_all_offers_cooled_falls_back_to_original_list(self) -> None:
        offers = [
            _make_offer(offer_id=1, machine_id=100),
            _make_offer(offer_id=2, machine_id=200),
        ]
        store = AsyncMock()
        store.cooled_machine_ids.return_value = {100, 200}

        result = await apply_cooldown_filter(offers, store)

        assert [o.id for o in result] == [1, 2]

    async def test_nothing_cooled_returns_equal_list(self) -> None:
        offers = [
            _make_offer(offer_id=1, machine_id=100),
            _make_offer(offer_id=2, machine_id=200),
        ]
        store = AsyncMock()
        store.cooled_machine_ids.return_value = set()

        result = await apply_cooldown_filter(offers, store)

        assert [o.id for o in result] == [1, 2]
