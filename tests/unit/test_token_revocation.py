"""Unit tests for TokenRevocationService (issue #142, rounds 1-3)."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from redis import exceptions as redis_exceptions
from redis.exceptions import NoScriptError

from src.api.services.token_revocation import (
    _EPOCH_WRITE_SCRIPT,
    TokenRevocationService,
    _epoch_key,
    _jti_key,
)

pytestmark = pytest.mark.unit

# _epoch_key normalizes through UUID(...) (A1), so fake payload subjects must
# be valid UUID strings, matching what create_access_token actually stamps.
_USER_ID = uuid4()
_USER_SUB = str(_USER_ID)


@dataclass
class _FakePayload:
    """Minimal stand-in for TokenPayload — only the fields is_revoked reads."""

    sub: str
    iat: int
    jti: str


def _make_redis() -> MagicMock:
    """A redis mock whose evalsha always raises NoScriptError, exercising the
    EVAL fallback path by default (see TestRevokeUserSessions for the
    direct-evalsha-hit path)."""
    redis = MagicMock()
    redis.set = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    redis.mget = AsyncMock(return_value=[None, None])
    redis.evalsha = AsyncMock(side_effect=NoScriptError("no cached script"))
    redis.eval = AsyncMock(return_value=1_700_000_000)
    return redis


class TestNoOpWhenRedisUnset:
    """redis_client=None must make every method a documented no-op."""

    async def test_revoke_user_sessions_noop(self) -> None:
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        result = await service.revoke_user_sessions(MagicMock())
        assert result is None

    async def test_revoke_token_noop_returns_true(self) -> None:
        """F5 — a documented no-op is not a failure, so it returns True."""
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        result = await service.revoke_token("some-jti", int(time.time()) + 60)
        assert result is True

    async def test_is_revoked_returns_false(self) -> None:
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        payload = _FakePayload(sub=_USER_SUB, iat=0, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False

    async def test_get_current_epoch_noop(self) -> None:
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        assert await service.get_current_epoch(uuid4()) is None

    def test_enabled_is_false(self) -> None:
        service = TokenRevocationService(None, max_token_ttl_seconds=900)
        assert service.enabled is False


class TestEpochKeyHelper:
    """A1 — the epoch key format must be pinned by a single helper shared by
    the write path (UUID) and read path (str), or bulk revocation silently
    stops matching with no error."""

    def test_uuid_and_str_produce_identical_key(self) -> None:
        user_id = uuid4()
        assert _epoch_key(user_id) == _epoch_key(str(user_id))

    async def test_key_written_by_revoke_matches_key_read_by_is_revoked(self) -> None:
        """End-to-end: the exact key revoke_user_sessions(uid) writes must be
        the exact key is_revoked reads for a token minted for uid."""
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        user_id = uuid4()

        await service.revoke_user_sessions(user_id)
        write_key = redis.eval.call_args.args[2]

        redis.mget.return_value = [None, None]
        payload = _FakePayload(sub=str(user_id), iat=0, jti="jti-1")
        await service.is_revoked(payload)  # type: ignore[arg-type]
        read_key = redis.mget.call_args.args[0][0]

        assert write_key == read_key == _epoch_key(user_id)


class TestRevokeUserSessions:
    """D1a/F1b — bulk revocation epoch written from Redis's own clock."""

    async def test_writes_epoch_via_eval_fallback_and_returns_it(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        user_id = uuid4()

        result = await service.revoke_user_sessions(user_id)

        assert result == 1_700_000_000
        redis.evalsha.assert_awaited_once()
        redis.eval.assert_awaited_once_with(_EPOCH_WRITE_SCRIPT, 1, _epoch_key(user_id), 1200)

    async def test_uses_evalsha_directly_when_script_is_cached(self) -> None:
        redis = _make_redis()
        redis.evalsha = AsyncMock(return_value=1_700_000_042)
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        user_id = uuid4()

        result = await service.revoke_user_sessions(user_id)

        assert result == 1_700_000_042
        redis.evalsha.assert_awaited_once()
        redis.eval.assert_not_called()

    async def test_redis_error_is_swallowed_and_returns_none(self) -> None:
        redis = _make_redis()
        redis.evalsha = AsyncMock(side_effect=RuntimeError("redis down"))
        redis.eval = AsyncMock(side_effect=RuntimeError("redis down"))
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        result = await service.revoke_user_sessions(uuid4())

        assert result is None

    async def test_failure_increments_failed_write_count(self) -> None:
        redis = _make_redis()
        redis.evalsha = AsyncMock(side_effect=RuntimeError("redis down"))
        redis.eval = AsyncMock(side_effect=RuntimeError("redis down"))
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        await service.revoke_user_sessions(uuid4())

        assert service.failed_write_count == 1


class TestGetCurrentEpoch:
    """F2 — read-only epoch lookup used by AuthService's post-mint race check."""

    async def test_returns_parsed_epoch(self) -> None:
        redis = _make_redis()
        redis.get = AsyncMock(return_value="1700000000")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        result = await service.get_current_epoch(uuid4())

        assert result == 1700000000

    async def test_returns_none_when_no_key(self) -> None:
        redis = _make_redis()
        redis.get = AsyncMock(return_value=None)
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        assert await service.get_current_epoch(uuid4()) is None

    async def test_redis_error_returns_none(self) -> None:
        redis = _make_redis()
        redis.get = AsyncMock(side_effect=RuntimeError("redis down"))
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        assert await service.get_current_epoch(uuid4()) is None


class TestRevokeToken:
    """D1b — per-jti denylist."""

    async def test_sets_ttl_from_remaining_lifetime(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        expires_at = int(time.time()) + 100

        result = await service.revoke_token("jti-abc", expires_at)

        assert result is True
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

        result = await service.revoke_token("jti-abc", expired_at)

        assert result is True
        redis.set.assert_not_called()

    async def test_redis_error_returns_false(self) -> None:
        redis = _make_redis()
        redis.set.side_effect = RuntimeError("redis down")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        result = await service.revoke_token("jti-abc", int(time.time()) + 60)

        assert result is False

    async def test_failure_increments_failed_write_count(self) -> None:
        redis = _make_redis()
        redis.set.side_effect = RuntimeError("redis down")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)

        await service.revoke_token("jti-abc", int(time.time()) + 60)

        assert service.failed_write_count == 1

    async def test_ttl_clamped_to_access_token_lifetime_under_skew(self) -> None:
        """A3 — a short computed ttl (simulating clock skew making the
        remaining lifetime look shorter than it really is) is clamped up to
        the configured access-token lifetime."""
        redis = _make_redis()
        service = TokenRevocationService(
            redis, max_token_ttl_seconds=1200, access_token_lifetime_seconds=900
        )
        expires_at = int(time.time()) + 10  # looks like only 10s left

        await service.revoke_token("jti-abc", expires_at)

        _args, kwargs = redis.set.call_args
        assert kwargs["ex"] == 900

    async def test_ttl_not_clamped_when_naturally_longer(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(
            redis, max_token_ttl_seconds=1200, access_token_lifetime_seconds=60
        )
        expires_at = int(time.time()) + 500

        await service.revoke_token("jti-abc", expires_at)

        _args, kwargs = redis.set.call_args
        assert 495 <= kwargs["ex"] <= 500


class TestIsRevoked:
    """Single round-trip; epoch hit, jti hit, neither, D2 (<=), F6, breaker."""

    async def test_false_when_neither_key_set(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = [None, None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False
        redis.mget.assert_awaited_once_with([_epoch_key(_USER_ID), _jti_key("jti-1")])

    async def test_true_on_jti_hit(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = [None, "1"]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is True

    async def test_true_when_iat_predates_epoch(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = ["1000", None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=999, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is True

    async def test_true_when_iat_equals_epoch(self) -> None:
        """D2 (round 2, F1) — `<=`, not `<`: a token minted in the exact
        same second as the revocation is rejected. This reverses round 1's
        design; do not "fix" it back to `<`."""
        redis = _make_redis()
        redis.mget.return_value = ["1000", None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is True

    async def test_false_when_iat_after_epoch(self) -> None:
        redis = _make_redis()
        redis.mget.return_value = ["1000", None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1001, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False

    async def test_single_round_trip(self) -> None:
        """Exactly one Redis call regardless of which key would match."""
        redis = _make_redis()
        redis.mget.return_value = ["1", "1"]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        await service.is_revoked(payload)  # type: ignore[arg-type]

        redis.mget.assert_awaited_once()
        redis.set.assert_not_called()

    async def test_malformed_epoch_fails_open_and_logs(self) -> None:
        """F6 — a garbage epoch value must not raise; the request succeeds
        (token treated as not revoked) and the event is logged."""
        redis = _make_redis()
        redis.mget.return_value = ["not-a-number", None]
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        with patch("src.api.services.token_revocation.logger") as mock_logger:
            result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False
        mock_logger.exception.assert_called_once()
        assert mock_logger.exception.call_args.args[0] == "authrev.malformed_epoch"

    async def test_redis_error_fails_open(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = RuntimeError("redis down")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False


class TestCircuitBreaker:
    """F4 — is_revoked's Redis calls are gated by an in-process breaker.

    Uses ``redis_exceptions.ConnectionError`` throughout as the stand-in
    failure — the "immediate" class (G3b), threshold 2 — for the general
    breaker mechanics (opening, logging, recovery). See
    TestBreakerThresholdByFailureClass below for the ConnectionError-vs-
    TimeoutError split itself.
    """

    async def test_breaker_stays_closed_under_threshold(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        await service.is_revoked(payload)  # type: ignore[arg-type]

        assert service.circuit_open is False
        assert redis.mget.await_count == 1

    async def test_breaker_opens_after_threshold_and_short_circuits(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        for _ in range(2):
            await service.is_revoked(payload)  # type: ignore[arg-type]
        assert service.circuit_open is True

        # A 3rd call must not touch Redis at all — breaker short-circuits.
        result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False
        assert redis.mget.await_count == 2

    async def test_breaker_logs_exactly_one_transition_line_during_outage(self) -> None:
        """Acceptance criterion (F4): one log line for the whole outage, not
        one per request."""
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        with patch("src.api.services.token_revocation.logger") as mock_logger:
            for _ in range(10):
                await service.is_revoked(payload)  # type: ignore[arg-type]

        assert mock_logger.error.call_count == 1
        assert mock_logger.error.call_args.args[0] == "authrev.circuit_opened"

    async def test_breaker_recovers_after_window_and_logs_once(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        for _ in range(2):
            await service.is_revoked(payload)  # type: ignore[arg-type]
        assert service.circuit_open is True

        # Force the breaker's window to have elapsed.
        service._breaker._opened_at = time.monotonic() - 10

        redis.mget.side_effect = None
        redis.mget.return_value = [None, None]

        with patch("src.api.services.token_revocation.logger") as mock_logger:
            result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False
        assert service.circuit_open is False
        mock_logger.info.assert_called_once()
        assert mock_logger.info.call_args.args[0] == "authrev.circuit_closed"


class TestBreakerThresholdByFailureClass:
    """G3b — trip threshold is picked per-failure by exception class.

    Immediate (``ConnectionError`` — refused/reset/DNS failure): 2.
    Ambiguous (``TimeoutError`` — slow/hung): 3. redis-py's ``TimeoutError``
    is a *subclass* of ``ConnectionError``, so this is also the regression
    test for the isinstance-ordering bug: testing ``ConnectionError`` first
    would misclassify every timeout as immediate and trip after 2.
    """

    async def test_connection_error_trips_after_two(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        await service.is_revoked(payload)  # type: ignore[arg-type]
        assert service.circuit_open is False

        await service.is_revoked(payload)  # type: ignore[arg-type]
        assert service.circuit_open is True

    async def test_timeout_error_does_not_trip_after_two(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.TimeoutError("slow")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        for _ in range(2):
            await service.is_revoked(payload)  # type: ignore[arg-type]

        assert service.circuit_open is False

    async def test_timeout_error_trips_after_three(self) -> None:
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.TimeoutError("slow")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        for _ in range(3):
            await service.is_revoked(payload)  # type: ignore[arg-type]

        assert service.circuit_open is True


class TestBreakerProbeReservation:
    """G3a — exactly one half-open probe is reserved per window, and
    circuit_open stays True while it's outstanding."""

    async def _trip_breaker(self, redis: MagicMock, service: TokenRevocationService) -> None:
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")
        for _ in range(2):
            await service.is_revoked(payload)  # type: ignore[arg-type]
        assert service.circuit_open is True
        # Force the window to have elapsed — now half-open.
        service._breaker._opened_at = time.monotonic() - 10

    async def test_only_one_probe_reaches_redis_under_concurrency(self) -> None:
        """A plain AsyncMock resolves without ever truly yielding to the
        event loop, so concurrent callers of it would run to completion
        one at a time rather than actually interleaving — which would prove
        nothing about the reservation. `_yielding_mget` forces a real
        suspension point (`asyncio.sleep(0)`) so the other 4 callers get a
        chance to call `allow_request()` while the probe caller is still
        mid-flight, genuinely exercising the reservation under concurrency.
        """
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        await self._trip_breaker(redis, service)
        calls_before_probe = redis.mget.await_count

        async def _yielding_mget(*_args: object, **_kwargs: object) -> list[None]:
            await asyncio.sleep(0)
            return [None, None]

        redis.mget.side_effect = _yielding_mget
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        results = await asyncio.gather(
            *(service.is_revoked(payload) for _ in range(5))  # type: ignore[arg-type]
        )

        # Exactly one of the 5 concurrent callers actually reached Redis —
        # the reserved probe. The other 4 were suppressed by allow_request.
        assert redis.mget.await_count == calls_before_probe + 1
        assert all(r is False for r in results)

    async def test_circuit_open_stays_true_while_probe_reserved(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        await self._trip_breaker(redis, service)

        # Simulate the probe having been claimed but not yet resolved.
        service._breaker._probe_in_flight = True

        assert service.circuit_open is True

    async def test_probe_slot_freed_after_probe_resolves(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        await self._trip_breaker(redis, service)

        redis.mget.side_effect = None
        redis.mget.return_value = [None, None]
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")
        await service.is_revoked(payload)  # type: ignore[arg-type]

        assert service.circuit_open is False
        assert service._breaker._probe_in_flight is False


class TestGetCurrentEpochBreakerGating:
    """B1 — get_current_epoch shares is_revoked's breaker instance."""

    async def test_breaker_open_short_circuits_without_touching_redis(self) -> None:
        redis = _make_redis()
        redis.get.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        user_id = uuid4()

        for _ in range(2):
            await service.get_current_epoch(user_id)
        assert service.circuit_open is True

        result = await service.get_current_epoch(user_id)

        assert result is None
        assert redis.get.await_count == 2

    async def test_shares_breaker_instance_with_is_revoked(self) -> None:
        """One refresh request's is_revoked + get_current_epoch pair
        together contribute to the same failure count — a single refresh
        can trip the breaker in one round trip instead of needing two."""
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        redis.get.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        await service.is_revoked(payload)  # type: ignore[arg-type]
        assert service.circuit_open is False

        await service.get_current_epoch(_USER_ID)

        assert service.circuit_open is True

    async def test_successful_call_resets_shared_breaker_state(self) -> None:
        """A successful get_current_epoch call resets the breaker state
        that is_revoked's earlier failure had started accumulating —
        confirms they share one counter, not two independent ones."""
        redis = _make_redis()
        redis.mget.side_effect = redis_exceptions.ConnectionError("refused")
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub=_USER_SUB, iat=1000, jti="jti-1")

        await service.is_revoked(payload)  # type: ignore[arg-type]
        assert service._breaker._consecutive_failures == 1

        redis.get.return_value = None
        await service.get_current_epoch(_USER_ID)

        assert service._breaker._consecutive_failures == 0
        assert service.circuit_open is False


class TestIsRevokedMalformedSubject:
    """B3 — is_revoked must not raise into a request on a non-UUID subject."""

    async def test_malformed_subject_fails_open_and_logs(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="not-a-uuid", iat=1000, jti="jti-1")

        with patch("src.api.services.token_revocation.logger") as mock_logger:
            result = await service.is_revoked(payload)  # type: ignore[arg-type]

        assert result is False
        mock_logger.exception.assert_called_once()
        assert mock_logger.exception.call_args.args[0] == "authrev.malformed_subject"

    async def test_malformed_subject_never_reaches_redis(self) -> None:
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="not-a-uuid", iat=1000, jti="jti-1")

        await service.is_revoked(payload)  # type: ignore[arg-type]

        redis.mget.assert_not_called()

    async def test_malformed_subject_does_not_count_toward_breaker(self) -> None:
        """An input error is not evidence Redis is down — it must not
        contribute to the breaker's failure threshold."""
        redis = _make_redis()
        service = TokenRevocationService(redis, max_token_ttl_seconds=1200)
        payload = _FakePayload(sub="not-a-uuid", iat=1000, jti="jti-1")

        for _ in range(5):
            await service.is_revoked(payload)  # type: ignore[arg-type]

        assert service.circuit_open is False
