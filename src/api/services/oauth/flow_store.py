"""Single-use OAuth flow state in Redis.

Every value is written with ``SET … EX`` and consumed with ``GETDEL`` (atomic
get-and-delete), so a state, handoff code or signup ticket can be redeemed at
most once even under concurrency. Only ``peek_signup`` reads without
consuming — signup-info must be repeatable while the user reads the legal
documents.

Uses the default short-lived Redis pool (``get_redis_client``). Redis errors
propagate: the OAuth flow fails closed rather than skipping single-use checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final, Protocol

import msgspec
import structlog

from src.api.services.oauth.models import OAuthFlow, OAuthHandoff, PendingSignup

if TYPE_CHECKING:
    from collections.abc import Callable

    from redis.asyncio import Redis

logger = structlog.get_logger(__name__)

FLOW_PREFIX: Final = "oauth:flow:"
HANDOFF_PREFIX: Final = "oauth:handoff:"
SIGNUP_PREFIX: Final = "oauth:signup:"


class OAuthFlowStore(Protocol):
    """Storage for the three kinds of single-use OAuth values."""

    async def put_flow(self, state: str, flow: OAuthFlow, ttl: int) -> None:
        """Store a pending authorize→callback flow."""
        ...

    async def take_flow(self, state: str) -> OAuthFlow | None:
        """Consume a flow (GETDEL)."""
        ...

    async def put_handoff(self, code: str, handoff: OAuthHandoff, ttl: int) -> None:
        """Store a one-time login handoff."""
        ...

    async def take_handoff(self, code: str) -> OAuthHandoff | None:
        """Consume a handoff (GETDEL)."""
        ...

    async def put_signup(self, ticket: str, signup: PendingSignup, ttl: int) -> None:
        """Store a pending signup."""
        ...

    async def peek_signup(self, ticket: str) -> PendingSignup | None:
        """Read a pending signup without consuming it (GET)."""
        ...

    async def take_signup(self, ticket: str) -> PendingSignup | None:
        """Consume a pending signup (GETDEL)."""
        ...


class RedisOAuthFlowStore:
    """``OAuthFlowStore`` over Redis (msgspec-JSON values).

    Args:
        client_factory: Returns a client on the default short-lived pool.
    """

    def __init__(self, client_factory: Callable[[], Redis]) -> None:
        self._client_factory = client_factory

    async def put_flow(self, state: str, flow: OAuthFlow, ttl: int) -> None:
        await self._put(FLOW_PREFIX + state, flow, ttl)

    async def take_flow(self, state: str) -> OAuthFlow | None:
        return _decode(await self._client_factory().getdel(FLOW_PREFIX + state), OAuthFlow)

    async def put_handoff(self, code: str, handoff: OAuthHandoff, ttl: int) -> None:
        await self._put(HANDOFF_PREFIX + code, handoff, ttl)

    async def take_handoff(self, code: str) -> OAuthHandoff | None:
        return _decode(await self._client_factory().getdel(HANDOFF_PREFIX + code), OAuthHandoff)

    async def put_signup(self, ticket: str, signup: PendingSignup, ttl: int) -> None:
        await self._put(SIGNUP_PREFIX + ticket, signup, ttl)

    async def peek_signup(self, ticket: str) -> PendingSignup | None:
        return _decode(await self._client_factory().get(SIGNUP_PREFIX + ticket), PendingSignup)

    async def take_signup(self, ticket: str) -> PendingSignup | None:
        return _decode(await self._client_factory().getdel(SIGNUP_PREFIX + ticket), PendingSignup)

    async def _put(self, key: str, value: msgspec.Struct, ttl: int) -> None:
        await self._client_factory().set(key, msgspec.json.encode(value), ex=ttl)


def _decode[T](raw: bytes | str | None, type_: type[T]) -> T | None:
    """Decode a stored value; ``bytes`` or ``str`` depending on the pool's decode setting."""
    if raw is None:
        return None
    data = raw.encode() if isinstance(raw, str) else bytes(raw)
    try:
        return msgspec.json.decode(data, type=type_)
    except msgspec.DecodeError:
        # Our own writes only — a corrupt value is treated as absent (the
        # flow then fails closed with flow_expired / invalid_*).
        logger.warning("auth.oauth.store_value_corrupt", value_type=type_.__name__)
        return None
