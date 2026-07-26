"""Access-token revocation — user epoch (bulk) + per-jti denylist (single-token).

Access tokens are stateless JWTs: the server never records issued `jti`s, so
there is no set of outstanding tokens to enumerate for a bulk revocation. This
service instead uses two O(1) Redis keys, both self-expiring:

  (a) User revocation epoch (``authrev:user:{user_id}``) — a unix-seconds
      timestamp written on every "revoke all sessions" event (logout-all,
      password change, account deactivation, password reset, refresh-token
      reuse detection). A token is rejected if its `iat` is at or before the
      epoch (``<=`` — see D2 below). TTL = the *maximum permitted* value of
      max(access token TTL, content cookie TTL), not the currently
      configured value (F3) — past that point no token issued before the
      epoch can still be valid, so the key is dead weight. One key per
      revocation *event*, not per token — this is what makes bulk revocation
      implementable without storing issued tokens.

  (b) Per-jti denylist (``authrev:jti:{jti}``) — written by single-device
      logout for the presenting access token's own `jti`, whose claims are
      already in hand at that point (nothing stored at issue time). TTL =
      remaining token lifetime, clamped to at least the configured
      access-token lifetime (A3) so minting-instance/denylisting-instance
      clock skew can never make the denylist entry expire before the token.

Both keys are read together in a single ``MGET`` round trip by ``is_revoked``.

D2 — ``<=``, not ``<`` (issue #142 round 2, F1): a prior revision of this
file used strict ``<`` on the theory that ``<=`` would reject a token minted
in the same wall-clock second as the revocation, breaking immediate
re-login. That reasoning was wrong and has been discarded — do not restore
it. ``iat <= epoch`` is arithmetically identical to ``iat < epoch + 1``; it
does not break re-login, it only refuses tokens minted inside the one-second
tick containing the revocation, after which login proceeds normally. The
client self-heals even in that tick: bulk revocation revokes refresh tokens
through a separate DB-backed path (see ``UserRepository.revoke_all_user_tokens``),
so a fresh login yields a valid refresh token regardless, and if the paired
access token happens to land in the rejected second, the existing 401->refresh
middleware mints a replacement one second later — invisible to the user.
Against that near-zero cost, strict ``<`` left an exploitable window
precisely where it matters most: refresh-token reuse detection, where an
attacker's freshly minted token plausibly shares the revocation's second.

D1b/F1b — the epoch is written using Redis's own clock (``TIME``), not the
API instance's local clock. ``iat`` is still stamped by the minting
instance at token-creation time — that side of the comparison cannot be
made clock-free without a token-contract change (see "rejected
alternative" below) — but the *write* side now has one authoritative clock
regardless of which of N API replicas handles the revoking request. Before
this, under multiple replicas, a minting instance running fast relative to
the revoking instance could issue a token whose local `iat` exceeded an
epoch written moments later by a slower instance — that token would survive
a revocation meant to kill it, with a window bounded by inter-instance clock
skew rather than by one second. Because `iat` is still local-clock in
origin, **NTP on all API hosts remains a hard deployment requirement** — see
``docs/CONFIGURATION.md``.

Considered and rejected — per-user generation counter: an alternative,
clock-free design embeds a monotonic per-user counter (``INCR``) as a `gen`
claim at mint time, with guards rejecting `payload.gen < stored_gen`. It
eliminates clock dependence entirely and is the textbook solution. Rejected
here as disproportionate: it requires a token contract change (a new claim
existing tokens lack, needing a rollout grace period), a Redis read on
*every* mint (today only guards read Redis; minting is Redis-free), and a
real answer for "Redis is unavailable at mint time" that the project's
existing fail-open posture has no good version of (mint an unusable token,
or block login entirely — both worse than the status quo). The project has
already accepted a bounded fail-open window during Redis outages (see D3
below); spending this much complexity to close a sub-second-plus-clock-skew
window is not a proportionate trade. Do not re-litigate this without first
re-reading this paragraph.

F4 — circuit breaker (round 3: G3a/G3b/B1 below): ``is_revoked`` and, since
B1, ``get_current_epoch`` run on every authenticated request and every
refresh respectively, so an unbounded per-request Redis timeout during an
outage would convert "fail open" into "fail slow" — every request pays the
full timeout before falling back. ``_RevocationCircuitBreaker``
(in-process, dependency-free, shared by both methods — B1) trips on a
class-dependent number of consecutive failures rather than a flat count:
2 for ``redis.exceptions.ConnectionError`` (refused/reset/DNS failure —
instantaneous and unambiguous evidence Redis is down) vs. 3 for
``redis.exceptions.TimeoutError`` (slow/hung — costs a full
``socket_timeout`` each and is ambiguous, since a GC pause or load spike
can produce one; tripping as eagerly as the immediate case would be a
security-relevant false-open from transient slowness). ``TimeoutError`` is
a *subclass* of ``ConnectionError`` in redis-py, so ``on_failure``'s
``isinstance`` check tests ``TimeoutError`` first — reversing that order
silently misclassifies every timeout as immediate and reintroduces the
false-open this split exists to avoid.

Once open, the breaker short-circuits for ``_BREAKER_OPEN_SECONDS`` (5s)
without touching Redis, then reserves exactly one half-open probe (G3a —
see ``allow_request``): once the window expires, one caller is let through
to test recovery and every other concurrent caller that same moment keeps
short-circuiting, rather than all of them issuing their own Redis call in
a thundering-herd burst. ``circuit_open`` stays True while that probe is
reserved, so the health surface (A2 below) doesn't flicker to healthy
mid-outage between the window expiring and the probe's result landing.

With ``socket_timeout=50ms`` and ``socket_connect_timeout=250ms``
(``Settings.redis_socket_timeout_seconds``/``redis_socket_connect_timeout_seconds``),
worst-case trip cost is ~0ms when Redis actively refuses/resets the
connection (2 x ~0ms), ~150ms when it's merely slow (3 x 50ms), and 500ms
in the rare black-holed-host case where connects hang until
``socket_connect_timeout`` (2 x 250ms) — down from a flat 750ms
(3 x 250ms) before this round. These are design-time estimates: measure
your actual Redis ``MGET`` p99 in a production-like environment before
relying on them, and raise ``socket_timeout`` if it's anywhere close to
50ms.

This bounds an outage's cost to one probe every 5 seconds instead of one
per request, and collapses what would otherwise be a per-request log line
into exactly one ``authrev.circuit_opened`` on the transition and one
``authrev.circuit_closed`` on recovery. Writes (``revoke_user_sessions``,
``revoke_token``) are rare (only on
logout/logout-all/password-change/reset/reuse-detection) and are
deliberately *not* behind the breaker — every write failure is logged,
since losing a security-relevant write is worth an operator's attention
every time.

B1 — ``get_current_epoch`` shares ``is_revoked``'s breaker instance
(issue #142 round 3): before this, an outage made every refresh pay a
second full Redis timeout on top of ``is_revoked``'s, and — because a
failed read already returned ``None`` — the F2 backstop below silently
disabled itself exactly when infrastructure was unhealthy, with no
distinguishing signal from "no epoch, nothing to do". Sharing the breaker
fixes both: the extra timeout is now short-circuited like any other Redis
call once the breaker is open, and one refresh request can now contribute
up to two failures toward the trip threshold, which trips the breaker
*faster* in practice during a real outage — desirable, and it composes
correctly with G3b's per-class thresholds since both methods report the
same exception types.

F5 — truthful reporting: ``revoke_user_sessions`` returns ``int | None`` (the
Redis-clock epoch that was written, or ``None`` on failure) and
``revoke_token`` returns ``bool`` (``True`` on success *or* a legitimate
no-op — Redis unset, or the token was already expired; ``False`` only on a
genuine Redis failure). Callers combine the return value with the
``enabled`` property to tell "Redis isn't configured" (expected, already
logged once at startup, not alert-worthy) apart from "Redis is configured
but this call failed" (an active security-relevant degradation, alert-worthy
via ``OpsEventBus``) — see call sites in ``AuthService``, ``UserService``,
and ``EmailVerificationService``.

F6 — malformed epoch values fail open, not closed: epoch parsing is fully
inside the same protected path as the Redis call, so a hand-edited or
legacy-format key logs ``authrev.malformed_epoch`` and is treated as "not
revoked" rather than raising ``ValueError`` into the caller's request (which
would 500 that user's every authenticated request — the opposite of this
service's documented fail-open posture).

A1 — key format is pinned by a single ``_epoch_key`` helper used by both the
write side (``revoke_user_sessions``, taking a ``UUID``) and the read side
(``is_revoked``/``get_current_epoch``, taking ``payload.sub``, a ``str``) so
the two can never silently diverge — see ``test_token_revocation.py`` for the
byte-identity test.

F2 — refresh-rotation race, now a backstop (issue #142 round 3, G1): the
primary guarantee lives in ``UserRepository.lock_user_for_session_change``
— a ``SELECT ... FOR UPDATE`` on the *user* row, acquired before either
the refresh-token row lock (``AuthService.refresh_tokens``, before
``get_refresh_token_by_hash_for_update``) or a bulk revocation's UPDATE
(``revoke_all_user_tokens``/``revoke_all_refresh_tokens``). See that
method's docstring for the full interleaving this closes and the required
lock ordering (user row -> refresh-token row in *every* path that
acquires both, or the two paths deadlock against each other). A row lock
alone couldn't close this gap: it only serializes against the *old*
refresh-token row, and can't protect a row that doesn't exist yet at the
time a bulk revocation's snapshot is taken — which is exactly the failure
mode the user-row lock fixes by serializing the two operations before
either touches the refresh_tokens table.

``get_current_epoch`` (read before minting, read again after — if the
epoch moved, the just-minted pair is revoked and the request refused) is
kept as a backstop for a revocation arriving via a path that somehow
bypasses the G1 lock. It is deliberately *not* relied upon as the primary
guarantee: as of B1 (below), a failed or breaker-suppressed read returns
``None`` the same as "no epoch key" — an outage during the mint window
silently disables this specific backstop rather than raising, which is an
acceptable trade only because the lock, not this check, is what actually
prevents the race.

A2 — the fail-open posture depends on an operator noticing Redis is down;
``enabled``, ``circuit_open``, and ``failed_write_count`` are surfaced
through ``TokenRevocationChecker`` (``src/api/services/health/checkers/
token_revocation.py``) into ``GET /v1/admin/health`` so that's visible on a
dashboard, not just in logs.

Fail-open, loudly (D3): ``redis_url`` defaults to ``None`` and Redis can be
transiently unavailable. Fail-closed is not acceptable here — it would
convert a Redis blip into a total authentication outage across the whole
product. Fail-open's blast radius is bounded by the access-token TTL (15 min
by default) and the content-cookie TTL, which is an acceptable trade. When
`redis_client` is `None` (``redis_url`` unset), every method is a documented
no-op — behaviour identical to pre-revocation Apex, not a silent gap. This
is a deliberate security posture — do not "fix" it into fail-closed.
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING
from uuid import UUID

import redis.exceptions
import structlog
from redis.exceptions import NoScriptError

if TYPE_CHECKING:
    import redis.asyncio as aioredis

    from src.api.security.jwt import TokenPayload

logger = structlog.get_logger(__name__)

_EPOCH_KEY_PREFIX = "authrev:user:"
_JTI_KEY_PREFIX = "authrev:jti:"

# F1b — writes the bulk-revocation epoch from Redis's own clock (TIME)
# rather than the local instance clock, so the write side has one
# authoritative source of truth regardless of which API replica handles the
# revoking request. Registered lazily: EVALSHA is attempted first using a
# locally computed SHA1 (identical to what SCRIPT LOAD would produce), with
# a plain EVAL fallback on NOSCRIPT (first call, or a flushed script cache).
_EPOCH_WRITE_SCRIPT = """
local t = redis.call('TIME')
redis.call('SET', KEYS[1], t[1], 'EX', ARGV[1])
return t[1]
"""
_EPOCH_WRITE_SCRIPT_SHA = hashlib.sha1(
    _EPOCH_WRITE_SCRIPT.encode(), usedforsecurity=False
).hexdigest()

# issue #142 G3b — the trip threshold is picked per-failure by exception
# class, not a single flat count: a refused/reset/DNS-failed connection
# (ConnectionError) is instantaneous and unambiguous evidence Redis is
# down, so 2 of them is enough to trip. A hung/slow Redis (TimeoutError)
# costs a full socket_timeout each and is ambiguous — a GC pause or load
# spike can produce one — so it takes 3 before tripping, to avoid a
# security-relevant false-open from transient slowness.
_BREAKER_IMMEDIATE_FAILURE_THRESHOLD = 2
_BREAKER_TIMEOUT_FAILURE_THRESHOLD = 3
_BREAKER_OPEN_SECONDS = 5.0


def _epoch_key(user_id: UUID | str) -> str:
    """Single source of truth for the bulk-revocation epoch key (A1).

    Both the write path (``revoke_user_sessions``, a ``UUID``) and the read
    paths (``is_revoked``/``get_current_epoch``, ``payload.sub`` — a JWT
    string subject) must produce byte-identical keys, or bulk revocation
    silently stops matching with no error anywhere. Normalises through
    ``UUID(...)`` so a string subject canonicalizes the same way a UUID
    object does, regardless of case/formatting.
    """
    return f"{_EPOCH_KEY_PREFIX}{UUID(str(user_id))}"


def _jti_key(jti: str) -> str:
    """Single source of truth for the per-jti denylist key."""
    return f"{_JTI_KEY_PREFIX}{jti}"


def _parse_epoch(raw: str | bytes | None, *, key: str) -> int | None:
    """F6 — parse defensively; a malformed value fails open, not closed.

    ``decode_responses=True`` means values always arrive as ``str`` in
    practice (the ``bytes`` branch is only in the type signature because
    redis-py's stubs keep it — ``int()`` accepts both), so a hand-edited or
    legacy-format key is reachable here. Returns ``None`` for "no key" and
    for "unparseable key" alike — both mean "cannot prove this token is
    revoked," which is the correct fail-open answer.
    """
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.exception("authrev.malformed_epoch", key=key, value=raw)
        return None


class _RevocationCircuitBreaker:
    """In-process, dependency-free breaker gating Redis calls (F4, B1).

    Not distributed — each process tracks its own failure count, which is
    fine: the goal is bounding *this process's* per-request latency during
    an outage, not cluster-wide coordination. Shared by both ``is_revoked``
    and ``get_current_epoch`` (issue #142 B1) — the failure mode is
    identical for both, and per-method breakers would fragment the signal
    (and the trip decision) across two independent counters for what is,
    from the operator's perspective, one outage.

    After a class-dependent number of consecutive failures (G3b —
    ``_BREAKER_IMMEDIATE_FAILURE_THRESHOLD``/``_BREAKER_TIMEOUT_FAILURE_THRESHOLD``)
    the breaker opens for ``_BREAKER_OPEN_SECONDS``, short-circuiting every
    call without touching Redis; after that window, exactly one probe is
    let through to test recovery (G3a — see ``allow_request``). A failed
    probe silently re-opens the window (no duplicate log) so a sustained
    outage produces exactly one ``authrev.circuit_opened`` line, not one
    per retry cycle.
    """

    def __init__(self) -> None:
        self._consecutive_failures = 0
        self._opened_at: float | None = None
        self._suppressed = 0
        self._probe_in_flight = False

    @property
    def is_open(self) -> bool:
        """True while short-circuiting: within the open window, or while a
        half-open probe is reserved (G3a). The latter case matters for the
        G2 health surface: without it, ``circuit_open`` would flicker to
        False the instant the window expires, even though every non-probe
        caller is still being short-circuited pending that probe's result.
        """
        if self._opened_at is None:
            return False
        if time.monotonic() - self._opened_at < _BREAKER_OPEN_SECONDS:
            return True
        return self._probe_in_flight

    def allow_request(self) -> bool:
        """False means: short-circuit, do not call Redis this time.

        G3a — reserves the single half-open probe: once the open window
        expires, exactly one caller is let through to test recovery, and
        every other concurrent caller in that same moment keeps
        short-circuiting rather than each issuing its own Redis call in a
        thundering-herd burst every ``_BREAKER_OPEN_SECONDS``.

        This method is synchronous and contains no ``await``, so under
        asyncio the check-and-reserve below cannot interleave with another
        call on the same event loop — that atomicity is a property of the
        event loop's cooperative scheduling, not of this code's shape. If
        this service is ever driven from a threaded context (multiple OS
        threads sharing one instance), this needs a real lock.
        """
        if self._opened_at is None:
            return True  # closed
        if time.monotonic() - self._opened_at < _BREAKER_OPEN_SECONDS:
            self._suppressed += 1
            return False  # open
        if self._probe_in_flight:
            self._suppressed += 1
            return False  # half-open, probe already claimed
        self._probe_in_flight = True
        return True  # half-open — this caller is the probe

    def on_success(self) -> None:
        if self._opened_at is not None:
            logger.info(
                "authrev.circuit_closed",
                op="is_revoked",
                suppressed_failures=self._suppressed,
            )
        self._consecutive_failures = 0
        self._opened_at = None
        self._suppressed = 0
        self._probe_in_flight = False

    def on_failure(self, exc: Exception) -> None:
        self._consecutive_failures += 1
        self._probe_in_flight = False
        # G3b — redis-py's TimeoutError is a SUBCLASS of ConnectionError,
        # so this isinstance check MUST test TimeoutError first. Reversing
        # the order would classify every timeout as an immediate failure,
        # tripping the breaker after just 2 slow responses and
        # reintroducing the false-open this threshold split exists to
        # avoid.
        threshold = (
            _BREAKER_TIMEOUT_FAILURE_THRESHOLD
            if isinstance(exc, redis.exceptions.TimeoutError)
            else _BREAKER_IMMEDIATE_FAILURE_THRESHOLD
        )
        if self._opened_at is None and self._consecutive_failures >= threshold:
            self._opened_at = time.monotonic()
            logger.error("authrev.circuit_opened", op="is_revoked", exc_info=exc)
        elif self._opened_at is not None:
            # Half-open probe failed — stay open, extend the window silently.
            self._opened_at = time.monotonic()


class TokenRevocationService:
    """Redis-backed access-token revocation. See module docstring for D1-D3, F1-F6, A1-A3."""

    def __init__(
        self,
        redis_client: aioredis.Redis | None,
        *,
        max_token_ttl_seconds: int,
        access_token_lifetime_seconds: int = 0,
    ) -> None:
        """Initialize the service.

        Args:
            redis_client: Redis client, or None to run as a documented no-op
                (mirrors the rest of the codebase — see rate_limit.py). Never
                calls get_redis_client() internally; the caller owns the
                connection's lifecycle.
            max_token_ttl_seconds: TTL for epoch keys — the *maximum
                permitted* value of max(access token TTL, content cookie
                TTL) per the current Settings field bounds (F3), not the
                currently configured value. Beyond this, no token issued
                before the epoch can still be valid.
            access_token_lifetime_seconds: Floor for the per-jti denylist
                TTL (A3) — the currently configured access-token lifetime,
                which upper-bounds any access token's true remaining
                lifetime at any point after mint. Defaults to 0 (no floor)
                so tests that don't exercise revoke_token's TTL math don't
                need to pass it.
        """
        self._redis = redis_client
        self._max_token_ttl_seconds = max_token_ttl_seconds
        self._access_token_lifetime_seconds = access_token_lifetime_seconds
        self._breaker = _RevocationCircuitBreaker()
        self._failed_write_count = 0
        if redis_client is None:
            logger.warning(
                "authrev.disabled",
                reason="redis_url not configured — revocation degrades to refresh-token-only",
            )

    @property
    def enabled(self) -> bool:
        """False when constructed with no Redis client (documented no-op)."""
        return self._redis is not None

    @property
    def circuit_open(self) -> bool:
        """True while the is_revoked breaker is short-circuiting Redis calls (A2)."""
        return self._breaker.is_open

    @property
    def failed_write_count(self) -> int:
        """Cumulative count of failed revoke_user_sessions/revoke_token calls (A2)."""
        return self._failed_write_count

    async def revoke_user_sessions(self, user_id: UUID) -> int | None:
        """Write a bulk-revocation epoch for the user (D1a), from Redis's clock (F1b).

        Any access or content token with `iat` at or before this epoch (D2)
        is rejected by guards from the next request onward.

        Returns:
            The epoch (Redis-clock unix seconds) that was written, or None
            if Redis is unset (documented no-op) or the write failed —
            callers use `enabled` to tell these apart (F5).
        """
        if self._redis is None:
            return None
        key = _epoch_key(user_id)
        try:
            try:
                raw_epoch = await self._redis.evalsha(
                    _EPOCH_WRITE_SCRIPT_SHA, 1, key, self._max_token_ttl_seconds
                )
            except NoScriptError:
                raw_epoch = await self._redis.eval(
                    _EPOCH_WRITE_SCRIPT, 1, key, self._max_token_ttl_seconds
                )
        except Exception:
            self._failed_write_count += 1
            logger.exception(
                "authrev.backend_unavailable",
                op="revoke_user_sessions",
                user_id=str(user_id),
            )
            return None
        return int(raw_epoch)

    async def get_current_epoch(self, user_id: UUID) -> int | None:
        """Read the current bulk-revocation epoch without writing (F2, now a backstop — see G1).

        Used by AuthService.refresh_tokens' post-mint race check: reads
        before and after minting a fresh token pair to detect a bulk
        revocation that landed via a path that somehow bypassed the G1
        user-row lock (``UserRepository.lock_user_for_session_change``),
        which is now the primary guarantee serializing rotation against a
        concurrent bulk revocation.

        Gated by the same circuit breaker instance as ``is_revoked``
        (issue #142 B1) — sharing it means one refresh request can
        contribute up to two failures toward the trip threshold (G3b),
        which is desirable: it makes the breaker trip faster in practice
        during an outage, not slower.

        Returns:
            The epoch, or None for "no epoch key", "Redis unset", "breaker
            open" (B1 — new), or "Redis failure" alike — the caller treats
            these identically (no diff observed => nothing to react to,
            consistent with fail-open). Before B1, a failed read already
            returned None here, which silently made the F2 race check a
            no-op during an outage — that is now an explicit, documented
            case rather than an accidental one.
        """
        if self._redis is None:
            return None
        if not self._breaker.allow_request():
            return None
        key = _epoch_key(user_id)
        try:
            raw = await self._redis.get(key)
        except Exception as exc:
            self._breaker.on_failure(exc)
            return None
        self._breaker.on_success()
        return _parse_epoch(raw, key=key)

    async def revoke_token(self, jti: str, expires_at_ts: int) -> bool:
        """Denylist a single token's jti until it would have expired anyway (D1b).

        Skips tokens that are already expired — nothing to gain from storing
        a key for a token that's already invalid.

        Returns:
            True on success or a legitimate no-op (Redis unset, or the token
            was already expired); False only on a genuine Redis failure (F5).
        """
        if self._redis is None:
            return True
        ttl = expires_at_ts - int(time.time())
        if ttl <= 0:
            return True
        # A3 — `expires_at_ts` was stamped by the minting instance; under
        # clock skew, this instance's clock can make the computed ttl come
        # out shorter than the token's true remaining lifetime, letting the
        # denylist entry expire before the token does. Clamp to the
        # configured access-token lifetime, which always upper-bounds any
        # access token's true remaining lifetime at any point after mint.
        ttl = max(ttl, self._access_token_lifetime_seconds)
        key = _jti_key(jti)
        try:
            await self._redis.set(key, "1", ex=ttl)
        except Exception:
            self._failed_write_count += 1
            logger.exception("authrev.backend_unavailable", op="revoke_token")
            return False
        return True

    async def is_revoked(self, payload: TokenPayload) -> bool:
        """Check both revocation mechanisms in a single Redis round-trip.

        Gated by a circuit breaker (F4) — during a sustained Redis outage,
        most calls short-circuit to False without touching Redis at all,
        bounding latency and log volume to one probe/log per breaker window
        rather than one per request.

        B3 — ``_epoch_key`` (which raises ``ValueError`` on a subject that
        isn't a valid UUID) is called from inside the protected block
        below, not before it. This method is a public entrypoint invoked
        on every authenticated request, and letting a malformed subject
        raise a 500 here would directly contradict this service's
        fail-open contract. Callers that already validate ``sub`` (the JWT
        guards, via ``_identity_from_token``) never reach this branch; it
        exists for callers that don't.

        Returns:
            True if the token's jti is denylisted or its `iat` is at or
            before the user's revocation epoch (D2 — `<=`). False on no
            match, on a malformed epoch (F6), on a malformed subject (B3),
            and — per the D3 fail-open posture — false while the breaker
            is open (including a reserved half-open probe, G3a) or on any
            Redis error.
        """
        if self._redis is None:
            return False
        if not self._breaker.allow_request():
            return False

        jti_key = _jti_key(payload.jti)
        try:
            epoch_key = _epoch_key(payload.sub)
            epoch_raw, jti_hit = await self._redis.mget([epoch_key, jti_key])
        except ValueError:
            logger.exception("authrev.malformed_subject", sub=payload.sub)
            return False
        except Exception as exc:
            self._breaker.on_failure(exc)
            return False
        self._breaker.on_success()

        if jti_hit is not None:
            return True

        epoch = _parse_epoch(epoch_raw, key=epoch_key)
        return False if epoch is None else payload.iat <= epoch
