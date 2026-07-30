"""Pre-commit revocation re-check for mutating endpoints (Claude board item).

Revocation (logout-all, password change/reset, deactivation, refresh-token
reuse detection) is enforced by ``auth_guard``/``content_auth_guard`` at the
*start* of a request. A mutating handler that authorizes work from that
token can still commit after the token is revoked mid-request — the guard
already passed before the revocation landed, and nothing re-checks
afterward. ``recheck_revocation_or_raise`` closes that window for the
handlers where it actually matters, by reusing the same
``lock_user_for_session_change`` + ``TokenRevocationService.is_revoked``
pattern first built for ``POST /v1/push/subscriptions``
(``src/api/routes/push.py``).

Which endpoints need this (vs. don't):

For most mutating endpoints this race is low-risk: they write data the
requesting user would legitimately own regardless of whether their session
was revoked a moment later (an upload or a generation job committed a few
milliseconds after a password change isn't a meaningful security gap, and
revoking access afterward removes any practical benefit anyway). This
helper is for the other kind — a write whose effect (a) outlives the
request, and (b) grants *continuing capability* to a party that isn't
necessarily "the same trusted session" anymore:

  - ``POST /v1/push/subscriptions`` — a subscription row keeps delivering
    push notifications to a device indefinitely; the whole point of
    "log out all devices, this one's compromised" is defeated if a
    subscription created in the race window survives it.
  - ``POST /v1/admin/manage/roles/{user_id}/grant`` and
    ``.../permissions/{user_id}/grant`` — grants a persistent privilege
    escalation. If a superadmin's own credentials are the reason their
    sessions are being revoked, an in-flight grant made with the
    now-revoked token must not still land.
  - ``POST /v1/organizations/{org_id}/members`` and
    ``PATCH .../members/{user_id}`` — grants persistent membership/role in
    an organization with a shared token-billing account.

Deliberately *not* wired into: ``revoke_role``/``revoke_permission``/
``remove_member`` (de-escalation — the safe direction, nothing to race),
``PATCH /v1/admin/users/{user_id}`` (can only set roles below ADMIN's own
guard rails and is a pre-existing, separately-scoped authorization
surface), GPU session start, billing top-ups, and ordinary library/storage
mutations (all "ordinary user-owned data" per the criteria above).

Cost: a `SELECT ... FOR UPDATE` on one already-fetched user row, only on
these low-traffic, security-relevant mutation endpoints — not on hot paths
like job creation or library edits. A cheaper lock-free check (compare
``get_current_epoch`` before/after) was considered and rejected as the
*primary* mechanism: without the lock, this write's commit can race a
revocation's snapshot in the other direction, observing a stale epoch — see
``TokenRevocationService`` F2, where the lock-free read is documented as a
backstop precisely because it can't close this gap alone.

Lock ordering / deadlock risk: this helper is frequently NOT the only lock
a handler takes. Any handler that writes a row referencing another user
also locks that user's row — explicitly (``UPDATE users``) or implicitly
(a ``FOR KEY SHARE`` foreign-key check on INSERT, e.g.
``organization_members.user_id`` or ``admin_permission_grants.user_id``).
Two requests with mirrored actor/target pairs then form a lock cycle.

Therefore: pass every user row the handler will touch via ``also_lock``, and
call this before any write. All of them are then acquired in one sorted
pass and no cycle can form. Locking one row (``push.py``, where actor ==
target) is the degenerate case, not the rule. The user-row ->
refresh-token-row ordering documented on
``UserRepository.lock_user_for_session_change`` still applies.

Fails open on Redis unavailability, same as every other revocation check
in this codebase (``TokenRevocationService`` D3): if ``is_revoked`` can't
reach Redis, the write proceeds rather than blocking on a cache outage.
``is_revoked`` reads both the bulk epoch and the per-jti denylist, and both
are now serialized by this lock: single-device logout's jti denylist write
(``AuthService.logout``) also takes ``lock_user_for_session_change`` before
writing to Redis, so the guarantee and the check no longer diverge — see
that method's docstring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from litestar.exceptions import NotAuthorizedException

from src.db.repositories import UserRepository

if TYPE_CHECKING:
    from collections.abc import Iterable
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.security.jwt import TokenPayload
    from src.api.services.token_revocation import TokenRevocationService


async def recheck_revocation_or_raise(
    *,
    session: AsyncSession,
    actor_id: UUID,
    token_payload: TokenPayload,
    token_revocation_service: TokenRevocationService,
    also_lock: Iterable[UUID] = (),
) -> None:
    """Lock every touched user row and re-check revocation before a durable write commits.

    Must be called before the handler's mutating write (and, transitively,
    before ``session.commit()``) — see the module docstring for why
    ordering and lock scope matter.

    Args:
        session: The request's transaction. The row locks live until this
            transaction commits or rolls back.
        actor_id: The user whose session might have been revoked — the
            caller of the endpoint, not necessarily the target of the
            write (e.g. the granting superadmin, not the user being
            granted a role).
        token_payload: The decoded access token the current request
            authenticated with (``get_current_token_payload``).
        token_revocation_service: Shared revocation checker.
        also_lock: Every other user row the handler will write or
            reference by foreign key — including rows locked implicitly by
            an FK check on INSERT (e.g. an ``organization_members`` or
            ``admin_permission_grants`` insert referencing ``user_id``).
            Locked together with ``actor_id`` in one sorted pass so mirrored
            actor/target requests can't form a deadlock cycle. Defaults to
            empty, which collapses to locking only ``actor_id`` (the
            ``push.py`` case, where actor == target).

    Raises:
        NotAuthorizedException: The token was revoked (bulk epoch or
            per-jti denylist) after the request started.
    """
    await UserRepository(session).lock_users_for_session_change((actor_id, *also_lock))
    if await token_revocation_service.is_revoked(token_payload):
        raise NotAuthorizedException(detail="Session has been revoked")
