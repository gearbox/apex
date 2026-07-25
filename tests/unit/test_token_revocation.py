"""Unit tests for TokenRevocationService (issue #142)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.api.services.token_revocation import TokenRevocationService

pytestmark = pytest.mark.unit


@dataclass
class _FakePayload:
    """Minimal stand-in for TokenPayload — only the fields is_revoked reads."""

    sub: str
    iat: int
    jti: str


def _make_redis() -> MagicMock:
    redis = MagicMock()
    redis.set = AsyncMock()
    redis.mget = AsyncMock(return_value=[None, None])
    return redis


class TestNoOpWhenRedisUnset:
    """redis_client=None must make every method a documented no-op."""

    async def test_revoke_user_sessions_noop(self) -> None:
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        await service.revoke_user_sessions(MagicMock())  # must not raise

    async def test_revoke_token_noop(self) -> None:
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        await service.revoke_token("some-jti", int(time.time()) + 60)  # must not raise

    async def test_is_revoked_returns_false(self) -> None:
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        payload = _FakePayload(sub="user-1", iat=0, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False


class TestRevokeUserSessions:
    """D1a — bulk revocation epoch."""

    async def test_writes_epoch_with_max_ttl(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        user_id = uuid4()

        before = int(time.time())
        await service.revoke_user_sessions(user_id)
        after = int(time.time())

        redis.set.assert_awaited_once()
        args, kwargs = redis.set.call_args
        assert args[0] == f"authrev:user:{user_id}"
        assert before <= args[1] <= after
        assert kwargs["ex"] == 1200

    async def test_redis_error_is_swallowed(self) -> None:
        redis = _make_redis()
        redis.set.side_effect = RuntimeError("redis down")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        await service.revoke_user_sessions(MagicMock())  # must not raise


class TestRevokeToken:
    """D1b — per-jti denylist."""

    async def test_sets_ttl_from_remaining_lifetime(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        expires_at = int(time.time()) + 100

        await service.revoke_token("jti-abc", expires_at)

        redis.set.assert_awaited_once()
        args, kwargs = redis.set.call_args
        assert args[0] == "authrev:jti:jti-abc"
        assert args[1] == "1"
        # ttl should be close to 100s (allow small scheduling slack)
        assert 95 <= kwargs["ex"] <= 100

    async def test_skips_already_expired_token(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        expired_at = int(time.time()) - 5

        await service.revoke_token("jti-abc", expired_at)

        redis.set.assert_not_called()

    async def test_redis_error_is_swallowed(self) -> None:
        redis = _make_redis()
        redis.set.side_effect = RuntimeError("redis down")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        await service.revoke_token("jti-abc", int(time.time()) + 60)  # must not raise


class TestIsRevoked:
    """Single round-trip; epoch hit, jti hit, neither, and fail-open on error."""

    async def test_false_when_neither_key_set(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = [None, None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="user-1", iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False
        redis.mget.assert_awaited_once_with(["authrev:user:user-1", "authrev:jti:jti-1"])

    async def test_true_on_jti_hit(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = [None, "1"]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="user-1", iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is True

    async def test_true_when_iat_predates_epoch(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = ["1000", None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="user-1", iat=999, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is True

    async def test_false_when_iat_equals_epoch(self) -> None:
        """D2 — strictly less-than, not <=: a token minted in the same second
        as the revocation must survive, or an immediate re-login after
        logout_all would be rejected."""
        redis = _make_redis()
        redis.mget.return_value = ["1000", None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="user-1", iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False

    async def test_false_when_iat_after_epoch(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = ["1000", None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="user-1", iat=1001, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False

    async def test_redis_error_fails_open(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = RuntimeError("redis down")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="user-1", iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False

    async def test_single_round_trip(self) -> None:
        """Exactly one Redis call regardless of which key would match."""
        redis = _make_redis()
        redis.mget.return_value = ["1", "1"]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="user-1", iat=1000, jti="jti-1")

        await service.is_revoked(payload)  # type: ignore[arg-type]

        redis.mget.assert_awaited_once()
        redis.set.assert_not_called()
