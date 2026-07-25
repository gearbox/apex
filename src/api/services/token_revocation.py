"""Access-token revocation — user epoch (bulk) + per-jti denylist (single-token).

Access tokens are stateless JWTs: the server never records issued `jti`s, so
there is no set of outstanding tokens to enumerate for a bulk revocation. This
service instead uses two O(1) Redis keys, both self-expiring:

  (a) User revocation epoch (``authrev:user:{user_id}``) — a unix-seconds
      timestamp written on every "revoke all sessions" event (logout-all,
      password change, account deactivation). A token is rejected if its
      `iat` predates the epoch. TTL = max(access token TTL, content cookie
      TTL) — past that point no token issued before the epoch can still be
      valid, so the key is dead weight. One key per revocation *event*, not
      per token — this is what makes bulk revocation implementable without
      storing issued tokens.

  (b) Per-jti denylist (``authrev:jti:{jti}``) — written by single-device
      logout for the presenting access token's own `jti`, whose claims are
      already in hand at that point (nothing stored at issue time). TTL =
      remaining token lifetime.

Fail-open, loudly (D3): ``redis_url`` defaults to ``None`` and Redis can be
transiently unavailable. Fail-closed is not acceptable here — it would
convert a Redis blip into a total authentication outage across the whole
product. Fail-open's blast radius is bounded by the access-token TTL (15 min
by default) and the content-cookie TTL, which is an acceptable trade. On any
Redis error this logs at `error` level under the ``authrev.backend_unavailable``
event and treats the token as not revoked. When `redis_client` is `None`
(``redis_url`` unset), every method is a documented no-op — behaviour
identical to pre-revocation Apex, not a silent gap. This is a deliberate
security posture — do not "fix" it into fail-closed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from uuid import UUID

    import redis.asyncio as aioredis

    from src.api.security.jwt import TokenPayload

logger = structlog.get_logger(__name__)

_EPOCH_KEY_PREFIX = "authrev:user:"
_JTI_KEY_PREFIX = "authrev:jti:"


class TokenRevocationService:
    """Redis-backed access-token revocation. See module docstring for D1-D3."""

    def __init__(
        self,
        redis_client: aioredis.Redis | None,
        *,
        max_token_ttl_seconds: int,
    ) -> None:
        """Initialize the service.

        Args:
            redis_client: Redis client, or None to run as a documented no-op
                (mirrors the rest of the codebase — see rate_limit.py). Never
                calls get_redis_client() internally; the caller owns the
                connection's lifecycle.
            max_token_ttl_seconds: TTL for epoch keys — max(access token TTL,
                content cookie TTL) in seconds. Beyond this, no token issued
                before the epoch can still be valid.
        """
        self._redis = redis_client
        self._max_token_ttl_seconds = max_token_ttl_seconds
        if redis_client is None:
            logger.warning(
                "authrev.disabled",
                reason="redis_url not configured — revocation degrades to refresh-token-only",
            )

    async def revoke_user_sessions(self, user_id: UUID) -> None:
        """Write a bulk-revocation epoch for the user (D1a).

        Any access or content token with `iat` before this call is rejected
        by guards from the next request onward.
        """
        if self._redis is None:
            return
        key = f"{_EPOCH_KEY_PREFIX}{user_id}"
        try:
            await self._redis.set(key, int(time.time()), ex=self._max_token_ttl_seconds)
        except Exception:
            logger.exception(
                "authrev.backend_unavailable",
                op="revoke_user_sessions",
                user_id=str(user_id),
            )

    async def revoke_token(self, jti: str, expires_at_ts: int) -> None:
        """Denylist a single token's jti until it would have expired anyway (D1b).

        Skips tokens that are already expired — nothing to gain from storing
        a key for a token that's already invalid.
        """
        if self._redis is None:
            return
        ttl = expires_at_ts - int(time.time())
        if ttl <= 0:
            return
        key = f"{_JTI_KEY_PREFIX}{jti}"
        try:
            await self._redis.set(key, "1", ex=ttl)
        except Exception:
            logger.exception("authrev.backend_unavailable", op="revoke_token")

    async def is_revoked(self, payload: TokenPayload) -> bool:
        """Check both revocation mechanisms in a single Redis round-trip.

        Returns:
            True if the token's jti is denylisted or its `iat` predates the
            user's revocation epoch. False on no match, and — per the D3
            fail-open posture — false on any Redis error.
        """
        if self._redis is None:
            return False
        epoch_key = f"{_EPOCH_KEY_PREFIX}{payload.sub}"
        jti_key = f"{_JTI_KEY_PREFIX}{payload.jti}"
        try:
            epoch_raw, jti_hit = await self._redis.mget([epoch_key, jti_key])
        except Exception:
            logger.exception("authrev.backend_unavailable", op="is_revoked")
            return False

        if jti_hit is not None:
            return True
        # D2: strictly less-than, not <=. iat has one-second granularity;
        # <= would kill a token minted in the same second as the
        # revocation, breaking immediate re-login right after
        # logout_all. < leaves a sub-second window where a token minted
        # in the same second as the revocation survives — negligible
        # against this threat model (the attacker already holds a
        # token; a <=1s issuance window changes nothing) and it keeps
        # re-login working. Do not change this to <=.
        return payload.iat < int(epoch_raw) if epoch_raw is not None else False
