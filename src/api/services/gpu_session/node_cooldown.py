"""Temporary cooldown list for broken Vast.ai machines.

Broken-but-verified machines are usually the cheapest offers, so they sort first
and get re-selected after every failure. This module records provisioning failures
per machine_id in Redis with an escalating TTL and filters cooled machines out of
offer candidates. Everything here is best-effort and fail-open: a Redis outage
must never block or fail provisioning.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

import structlog

if TYPE_CHECKING:
    from collections.abc import Sequence

    import redis.asyncio as aioredis

    from src.api.services.vastai.schemas import VastAIOffer
    from src.core.config import Settings

logger = structlog.get_logger(__name__)

_COOLDOWN_KEY = "vastai:node_cooldown:{machine_id}"
_FAILCOUNT_KEY = "vastai:node_failcount:{machine_id}"


class NodeCooldownStore(Protocol):
    """Interface for the broken-node cooldown store (DI seam)."""

    async def record_failure(self, machine_id: int, *, reason: str) -> None: ...

    async def cooled_machine_ids(self, machine_ids: Sequence[int]) -> set[int]: ...


class NullNodeCooldownStore:
    """No-op store used when Redis is not configured."""

    async def record_failure(self, machine_id: int, *, reason: str) -> None:  # noqa: ARG002
        return None

    async def cooled_machine_ids(self, machine_ids: Sequence[int]) -> set[int]:  # noqa: ARG002
        return set()


class RedisNodeCooldownStore:
    """Redis-backed cooldown store with escalating TTL.

    TTL = min(base * 2^(failure_count - 1), cap). The failure counter lives in its
    own key with a sliding window TTL, so machines that keep failing after each
    cooldown expiry get progressively longer cooldowns until Vast.ai deverifies
    them or the host fixes the machine.
    """

    def __init__(self, redis: aioredis.Redis, settings: Settings) -> None:
        self._redis = redis
        self._base_seconds = settings.node_cooldown_base_minutes * 60
        self._max_seconds = settings.node_cooldown_max_minutes * 60
        self._window_seconds = settings.node_cooldown_failure_window_hours * 3600

    async def record_failure(self, machine_id: int, *, reason: str) -> None:
        # Best-effort: intentionally broad except — cooldown must never break provisioning.
        try:
            failcount_key = _FAILCOUNT_KEY.format(machine_id=machine_id)
            pipe = self._redis.pipeline()
            pipe.incr(failcount_key)
            pipe.expire(failcount_key, self._window_seconds)
            count, _ = await pipe.execute()
            ttl = min(self._base_seconds * 2 ** (int(count) - 1), self._max_seconds)
            await self._redis.set(_COOLDOWN_KEY.format(machine_id=machine_id), reason, ex=ttl)
            logger.warning(
                "gpu_session.node_cooldown.recorded",
                machine_id=machine_id,
                failure_count=int(count),
                ttl_seconds=ttl,
                reason=reason,
            )
        except Exception:
            logger.exception("gpu_session.node_cooldown.record_failed", machine_id=machine_id)

    async def cooled_machine_ids(self, machine_ids: Sequence[int]) -> set[int]:
        if not machine_ids:
            return set()
        try:
            keys = [_COOLDOWN_KEY.format(machine_id=m) for m in machine_ids]
            values = await self._redis.mget(keys)
            return {m for m, v in zip(machine_ids, values, strict=True) if v is not None}
        except Exception:
            # Fail-open: on Redis trouble, behave as if nothing is cooled.
            logger.warning("gpu_session.node_cooldown.lookup_failed", exc_info=True)
            return set()


async def apply_cooldown_filter(
    offers: Sequence[VastAIOffer], store: NodeCooldownStore
) -> list[VastAIOffer]:
    """Drop offers on cooled machines, preserving order.

    Soft filter (D6): if every candidate is cooled, return the original list —
    a suspect node beats an artificial no-capacity failure. Offers without a
    machine_id always pass through (not identifiable → not filterable).
    """
    candidate_ids = [o.machine_id for o in offers if o.machine_id is not None]
    cooled = await store.cooled_machine_ids(candidate_ids)
    if not cooled:
        return list(offers)
    kept = [o for o in offers if o.machine_id is None or o.machine_id not in cooled]
    if not kept:
        logger.warning(
            "gpu_session.node_cooldown.all_offers_cooled",
            offer_count=len(offers),
            cooled_count=len(cooled),
        )
        return list(offers)
    logger.info(
        "gpu_session.node_cooldown.offers_filtered",
        excluded=len(offers) - len(kept),
        remaining=len(kept),
    )
    return kept
