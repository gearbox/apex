"""Shared in-memory Redis stand-in for token-revocation tests.

``FakeRedis`` implements the ``set``/``mget``/``get``/``eval``/``evalsha`` subset
``TokenRevocationService`` needs, with real TTL-expiry semantics and a pluggable
clock simulating Redis ``TIME`` — consistent with this repo's other infra-free
"integration" tests that exercise real guards/DI/HTTP without a live Redis.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock

from redis.exceptions import NoScriptError

from src.api.services.ops_event_bus import OpsEventBus
from src.api.services.session_termination import (
    SessionTerminationFactory,
    SessionTerminationService,
    make_session_termination_factory,
)
from src.api.services.token_revocation import TokenRevocationService

if TYPE_CHECKING:
    from collections.abc import Callable


class FakeRedis:
    """Minimal in-memory stand-in for redis.asyncio.Redis.

    ``eval`` simulates the production Lua epoch-write script
    (``redis.call('TIME')`` + ``SET ... EX``) using a pluggable clock so tests
    can pin the "Redis clock" deterministically instead of depending on
    real-clock timing.
    """

    def __init__(self, *, clock: Callable[[], float] = time.time) -> None:
        self._store: dict[str, tuple[str, float | None]] = {}
        self._clock = clock

    async def set(self, key: str, value: object, ex: int | None = None) -> None:
        deadline = self._clock() + ex if ex is not None else None
        self._store[key] = (str(value), deadline)

    async def mget(self, keys: list[str]) -> list[str | None]:
        now = self._clock()
        result: list[str | None] = []
        for key in keys:
            entry = self._store.get(key)
            if entry is None:
                result.append(None)
                continue
            value, deadline = entry
            if deadline is not None and deadline < now:
                del self._store[key]
                result.append(None)
            else:
                result.append(value)
        return result

    async def get(self, key: str) -> str | None:
        (result,) = await self.mget([key])
        return result

    async def evalsha(self, _sha: str, _numkeys: int, *_keys_and_args: object) -> int:
        raise NoScriptError("fake redis never has a cached script")

    async def eval(self, _script: str, numkeys: int, *keys_and_args: object) -> int:
        """Simulates the production epoch-write script: SET key=TIME, EX=ttl."""
        key = str(keys_and_args[0])
        ttl = int(str(keys_and_args[numkeys]))
        now = int(self._clock())
        await self.set(key, now, ex=ttl)
        return now


def make_session_termination(
    *,
    user_repo: Any,
    token_revocation: TokenRevocationService,
    session: Any | None = None,
    ops_event_bus: OpsEventBus | None = None,
) -> SessionTerminationService:
    """A real ``SessionTerminationService`` over the collaborators a test already has.

    ``AuthService``/``UserService`` require one (W1-C), so tests build it from the
    same repository / revocation service / ops bus / session they hand the service —
    the revoke-all sequence then runs against exactly the doubles the test asserts on.
    """
    return SessionTerminationService(
        session=session if session is not None else AsyncMock(),
        user_repo=user_repo,
        token_revocation=token_revocation,
        ops_event_bus=ops_event_bus if ops_event_bus is not None else OpsEventBus(enabled=False),
    )


def make_session_termination_factory_noop() -> SessionTerminationFactory:
    """A factory over a revocation service and ops bus that no-op (Redis unset, bus disabled)."""
    return make_session_termination_factory(
        token_revocation=TokenRevocationService(None, max_token_ttl_seconds=0),
        ops_event_bus=OpsEventBus(enabled=False),
    )
