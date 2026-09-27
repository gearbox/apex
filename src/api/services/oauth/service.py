"""OAuth login/signup orchestration (server-side OIDC authorization-code flow).

Request-scoped (wraps session-bound repositories) and **never commits** —
the callback route commits between :meth:`OAuthService.resolve_callback`
(DB writes) and :meth:`OAuthService.issue_redirect` (Redis handoff), so a
handoff code is never issued for a link that later rolls back. Savepoints
(``begin_nested``) are used only to turn unique violations into
``IdentityConflictError`` and to keep a failed signup all-or-nothing.

Never logs codes, state, tickets, the binding, the PKCE verifier, tokens,
the provider ``sub``, or the provider email.
"""

from __future__ import annotations

import hmac
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Final
from urllib.parse import quote, urlencode

import structlog
from sqlalchemy.exc import IntegrityError

from src.api.security.oauth_tx_cookie import binding_hash
from src.api.services.oauth.errors import (
    AccountExistsUnverifiedError,
    AccountInactiveError,
    FlowExpiredError,
    IdentityConflictError,
    InvalidHandoffError,
    InvalidSignupTicketError,
)
from src.api.services.oauth.models import (
    LoginOutcome,
    OAuthFlow,
    OAuthHandoff,
    PendingSignup,
    SignupOutcome,
    VerifiedIdentity,
)
from src.api.services.oauth.pkce import (
    generate_opaque_token,
    generate_verifier,
    s256_challenge,
    validate_return_to,
)
from src.core.enums import OAuthErrorCode, OAuthResult
from src.core.uid import new_id

if TYPE_CHECKING:
    from collections.abc import Sequence
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession

    from src.api.services.auth import AuthService, TokenPair
    from src.api.services.legal.acceptance import AcceptedDocument, RequestContext
    from src.api.services.oauth.flow_store import OAuthFlowStore
    from src.api.services.oauth.models import CallbackOutcome
    from src.api.services.oauth.registry import OAuthProviderRegistry
    from src.core.config import Settings
    from src.core.product import OAuthProvider, ProductConfig
    from src.db.models import User
    from src.db.repositories import UserRepository
    from src.db.repositories.user_identity import UserIdentityRepository

logger = structlog.get_logger(__name__)

_IDENTITY_CONSTRAINTS: Final = frozenset(
    {"uq_user_identities_product_provider_subject", "uq_user_identities_user_provider"}
)


class CallbackRejectReason(StrEnum):
    """Why a callback's flow state was refused (logged; the wire code is always flow_expired)."""

    UNKNOWN_STATE = "unknown_state"
    MISSING_BINDING = "missing_binding"
    BINDING_MISMATCH = "binding_mismatch"
    PRODUCT_MISMATCH = "product_mismatch"
    PROVIDER_MISMATCH = "provider_mismatch"


def _binding_matches(binding: str | None, expected_hash: str) -> bool:
    return binding is not None and hmac.compare_digest(
        binding_hash(binding).encode(), expected_hash.encode()
    )


def _is_identity_conflict(exc: IntegrityError) -> bool:
    """Whether a unique violation hit one of the two ``user_identities`` constraints."""
    cause: BaseException | None = exc.orig
    while cause is not None:
        if getattr(cause, "constraint_name", None) in _IDENTITY_CONSTRAINTS:
            return True
        cause = cause.__cause__
    return any(name in str(exc.orig) for name in _IDENTITY_CONSTRAINTS)


class OAuthService:
    """Drives authorize → callback → exchange / signup for every OAuth provider."""

    def __init__(
        self,
        *,
        registry: OAuthProviderRegistry,
        store: OAuthFlowStore,
        identity_repo: UserIdentityRepository,
        user_repo: UserRepository,
        auth_service: AuthService,
        session: AsyncSession,
        settings: Settings,
    ) -> None:
        self._registry = registry
        self._store = store
        self._identities = identity_repo
        self._users = user_repo
        self._auth = auth_service
        self._session = session
        self._settings = settings

    # ------------------------------------------------------------------
    # authorize
    # ------------------------------------------------------------------

    async def start(
        self,
        *,
        product: ProductConfig,
        provider: OAuthProvider,
        return_to: str | None,
        binding: str,
    ) -> str:
        """Persist a new flow and return the provider authorization URL.

        Raises:
            OAuthProviderNotEnabledError: Provider not allowed/configured (404).
            ValueError: ``return_to`` is not a same-origin path (400).
        """
        return_to = validate_return_to(return_to)
        client = self._registry.get(product, provider)
        state = generate_opaque_token()
        nonce = generate_opaque_token()
        verifier = generate_verifier()
        await self._store.put_flow(
            state,
            OAuthFlow(
                product_id=product.slug,
                provider=provider,
                code_verifier=verifier,
                nonce=nonce,
                binding_hash=binding_hash(binding),
                return_to=return_to,
            ),
            self._settings.oauth_flow_ttl_seconds,
        )
        logger.info("auth.oauth.started", product_id=product.slug, provider=provider.value)
        return client.authorization_url(
            redirect_uri=self._registry.redirect_uri(product, provider),
            state=state,
            nonce=nonce,
            code_challenge=s256_challenge(verifier),
        )

    # ------------------------------------------------------------------
    # callback
    # ------------------------------------------------------------------

    async def resolve_callback(
        self,
        *,
        product: ProductConfig,
        provider: OAuthProvider,
        state: str,
        code: str,
        binding: str | None,
    ) -> tuple[CallbackOutcome, str | None]:
        """Validate the flow, redeem the code, and resolve login vs. signup.

        Performs the identity link / ``last_login_at`` touch (DB) but no Redis
        writes. The caller must commit before :meth:`issue_redirect`.

        Returns:
            ``(outcome, return_to)``.

        Raises:
            FlowExpiredError: Unknown/consumed state, or binding/product/provider mismatch.
            OAuthFailedError: Token endpoint or id_token verification failed.
            EmailUnverifiedError: The provider did not verify the email.
            AccountInactiveError: The linked account is deactivated.
            AccountExistsUnverifiedError: Same-email local account is unverified.
            IdentityConflictError: The local account is linked to another subject.
        """
        # 1. Consume the flow first — a replayed state fails even if the rest would pass.
        flow = await self._store.take_flow(state)
        reason = self._flow_reject_reason(flow, product, provider, binding)
        if reason is not None or flow is None:
            logger.warning(
                "auth.oauth.callback_rejected",
                product_id=product.slug,
                provider=provider.value,
                reason=(reason or CallbackRejectReason.UNKNOWN_STATE).value,
            )
            raise FlowExpiredError

        # 2. Redeem the code (network) and verify the id_token.
        client = self._registry.get(product, provider)
        identity = await client.exchange_code(
            code=code,
            code_verifier=flow.code_verifier,
            redirect_uri=self._registry.redirect_uri(product, provider),
            nonce=flow.nonce,
        )

        # 3. Known identity → login (unless the account was deactivated).
        linked = await self._identities.get_by_subject(
            product_id=product.slug, provider=provider, subject=identity.subject
        )
        if linked is not None:
            user = await self._users.get_user(linked.user_id)
            if user is None or not user.is_active:
                logger.info(
                    "auth.oauth.login_rejected_inactive",
                    user_id=str(linked.user_id),
                    provider=provider.value,
                )
                raise AccountInactiveError
            await self._identities.touch_last_login(linked.id)
            logger.info("auth.oauth.login_resolved", user_id=str(user.id), provider=provider.value)
            return LoginOutcome(user_id=user.id), flow.return_to

        # 4. Same-email local account → auto-link only if both sides verified.
        local = await self._users.get_active_user_by_email(identity.email, product_id=product.slug)
        if local is not None:
            await self._link_existing(local, identity, product)
            return LoginOutcome(user_id=local.id), flow.return_to

        # 5. Unknown identity → two-step signup. No DB writes.
        logger.info("auth.oauth.signup_pending", product_id=product.slug, provider=provider.value)
        return SignupOutcome(identity=identity), flow.return_to

    @staticmethod
    def _flow_reject_reason(
        flow: OAuthFlow | None,
        product: ProductConfig,
        provider: OAuthProvider,
        binding: str | None,
    ) -> CallbackRejectReason | None:
        if flow is None:
            return CallbackRejectReason.UNKNOWN_STATE
        if binding is None:
            return CallbackRejectReason.MISSING_BINDING
        if not _binding_matches(binding, flow.binding_hash):
            return CallbackRejectReason.BINDING_MISMATCH
        if flow.product_id != product.slug:
            return CallbackRejectReason.PRODUCT_MISMATCH
        if flow.provider != provider:
            return CallbackRejectReason.PROVIDER_MISMATCH
        return None

    async def _link_existing(
        self, local: User, identity: VerifiedIdentity, product: ProductConfig
    ) -> None:
        if local.email_verified_at is None:
            # Pre-account hijacking guard: someone could have registered this
            # email without owning the inbox. Recovery: verify or reset password.
            logger.info("auth.oauth.link_rejected_unverified", user_id=str(local.id))
            raise AccountExistsUnverifiedError
        if await self._identities.get_for_user(user_id=local.id, provider=identity.provider):
            logger.warning(
                "auth.oauth.link_rejected_conflict",
                user_id=str(local.id),
                provider=identity.provider.value,
            )
            raise IdentityConflictError
        linked = await self._add_identity(local.id, identity, product)
        await self._identities.touch_last_login(linked)
        logger.info(
            "auth.oauth.identity_linked", user_id=str(local.id), provider=identity.provider.value
        )

    async def _add_identity(
        self, user_id: UUID, identity: VerifiedIdentity, product: ProductConfig
    ) -> UUID:
        """Insert an identity in a savepoint; a unique violation → ``IdentityConflictError``."""
        identity_id = new_id()
        try:
            # Releasing the savepoint flushes the INSERT, so a unique violation
            # surfaces here and rolls back only this savepoint.
            async with self._session.begin_nested():
                self._identities.add(
                    id=identity_id,
                    user_id=user_id,
                    product_id=product.slug,
                    provider=identity.provider,
                    subject=identity.subject,
                )
        except IntegrityError as exc:
            if _is_identity_conflict(exc):
                logger.warning(
                    "auth.oauth.identity_conflict",
                    user_id=str(user_id),
                    provider=identity.provider.value,
                )
                raise IdentityConflictError from exc
            raise
        return identity_id

    async def issue_redirect(
        self,
        *,
        product: ProductConfig,
        outcome: CallbackOutcome,
        return_to: str | None,
        binding: str,
    ) -> str:
        """Write the handoff/ticket to Redis and build the frontend fragment URL.

        Call only after the callback's DB writes are committed.
        """
        params: dict[str, str]
        match outcome:
            case LoginOutcome(user_id=user_id):
                code = generate_opaque_token()
                await self._store.put_handoff(
                    code,
                    OAuthHandoff(
                        product_id=product.slug, user_id=user_id, binding_hash=binding_hash(binding)
                    ),
                    self._settings.oauth_handoff_ttl_seconds,
                )
                params = {"result": OAuthResult.LOGIN.value, "code": code}
            case SignupOutcome(identity=identity):
                ticket = generate_opaque_token()
                await self._store.put_signup(
                    ticket,
                    PendingSignup(
                        product_id=product.slug,
                        provider=identity.provider,
                        subject=identity.subject,
                        email=identity.email,
                        binding_hash=binding_hash(binding),
                    ),
                    self._settings.oauth_signup_ticket_ttl_seconds,
                )
                params = {"result": OAuthResult.SIGNUP.value, "ticket": ticket}
        if return_to is not None:
            params["return_to"] = return_to
        return self._frontend_url(product, params)

    def error_redirect(self, *, product: ProductConfig, error: OAuthErrorCode) -> str:
        """Frontend fragment URL reporting a flow error."""
        return self._frontend_url(
            product, {"result": OAuthResult.ERROR.value, "error": error.value}
        )

    def _frontend_url(self, product: ProductConfig, params: dict[str, str]) -> str:
        # The fragment never reaches any server (no Referer/log leak); the SPA
        # strips it with history.replaceState before making requests.
        base = self._settings.app_url_for(product.slug).rstrip("/")
        fragment = urlencode(params, quote_via=quote, safe="")
        return f"{base}{self._settings.oauth_frontend_callback_path}#{fragment}"

    # ------------------------------------------------------------------
    # exchange / signup
    # ------------------------------------------------------------------

    async def exchange(
        self,
        *,
        product: ProductConfig,
        code: str,
        binding: str | None,
        context: RequestContext,
    ) -> tuple[UUID, TokenPair]:
        """Redeem a login handoff code for a token pair (first credential mint).

        Raises:
            InvalidHandoffError: Unknown/consumed/expired code, or binding/product mismatch.
            AccountInactiveError: The account was deactivated since the callback.
        """
        handoff = await self._store.take_handoff(code)
        if (
            handoff is None
            or not _binding_matches(binding, handoff.binding_hash)
            or handoff.product_id != product.slug
        ):
            logger.warning("auth.oauth.exchange_rejected", product_id=product.slug)
            raise InvalidHandoffError
        user = await self._users.get_active_user(handoff.user_id)
        if user is None:
            raise AccountInactiveError
        tokens = await self._auth.issue_session(user.id, product_id=product.slug, context=context)
        logger.info("auth.oauth.login_completed", user_id=str(user.id))
        return user.id, tokens

    async def signup_info(
        self, *, product: ProductConfig, ticket: str, binding: str | None
    ) -> PendingSignup:
        """Read a pending signup (non-consuming) for the signup screen.

        Raises:
            InvalidSignupTicketError: Unknown/expired ticket, or binding/product mismatch.
        """
        return await self._peek_valid_signup(product, ticket, binding)

    async def complete_signup(
        self,
        *,
        product: ProductConfig,
        ticket: str,
        binding: str | None,
        accepted_documents: Sequence[AcceptedDocument],
        display_name: str | None,
        context: RequestContext,
    ) -> tuple[User, TokenPair]:
        """Create the user + identity + billing account + SIGNUP acceptances, then mint tokens.

        Legal validation runs before the ticket is consumed, so a stale form
        (409/422) can be retried with the current documents.

        Raises:
            InvalidSignupTicketError: Invalid ticket, or lost the race for it.
            LegalSubmissionIncompleteError: Required documents missing/extra (422).
            LegalVersionStaleError: A submitted version is not current (409).
            EmailAlreadyExistsError: The email was registered since the callback.
            IdentityConflictError: The subject was linked since the callback.
        """
        # 1. Non-consuming read + ownership checks.
        await self._peek_valid_signup(product, ticket, binding)
        # 2. Pure legal validation — raises before the ticket is consumed.
        validated = self._auth.validate_signup(product.slug, accepted_documents)
        # 3. Consume. Exactly one concurrent caller gets the value.
        pending = await self._store.take_signup(ticket)
        if pending is None:
            logger.info("auth.oauth.signup_ticket_race_lost", product_id=product.slug)
            raise InvalidSignupTicketError

        identity = _pending_identity(pending)
        user_id = new_id()
        # 4-5. All-or-nothing: an identity conflict must not leave a user behind.
        async with self._session.begin_nested():
            user = await self._auth.provision_user(
                user_id=user_id,
                email=pending.email,
                password_hash=None,
                validated=validated,
                context=context,
                display_name=display_name,
                email_verified_at=datetime.now(UTC),
            )
            await self._add_identity(user_id, identity, product)

        # 6. First credential mint — carries the lgl digest just earned.
        tokens = await self._auth.issue_session(user_id, product_id=product.slug, context=context)
        logger.info(
            "auth.oauth.signup_completed", user_id=str(user_id), provider=pending.provider.value
        )
        return user, tokens

    async def _peek_valid_signup(
        self, product: ProductConfig, ticket: str, binding: str | None
    ) -> PendingSignup:
        pending = await self._store.peek_signup(ticket)
        if (
            pending is None
            or not _binding_matches(binding, pending.binding_hash)
            or pending.product_id != product.slug
        ):
            logger.warning("auth.oauth.signup_ticket_rejected", product_id=product.slug)
            raise InvalidSignupTicketError
        return pending


def _pending_identity(pending: PendingSignup) -> VerifiedIdentity:
    return VerifiedIdentity(provider=pending.provider, subject=pending.subject, email=pending.email)
