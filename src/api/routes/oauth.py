"""OAuth login/signup routes (server-side OIDC authorization-code flow).

Unguarded by design: these endpoints *establish* a session. Browser binding
is the ``apex_oauth_tx`` cookie (checked by the service at every step), and
tokens are minted only at ``/exchange`` and ``/complete-signup``.
Canonical frontend contract: docs/contracts/oauth-contract.md.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, Final, cast

import structlog
from litestar import Controller, Response, get, post
from litestar.exceptions import NotFoundException
from litestar.params import Body, Parameter
from litestar.response import Redirect
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_302_FOUND,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
    HTTP_409_CONFLICT,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.routes._token_response import build_token_response
from src.api.schemas.auth import TokenResponse
from src.api.schemas.errors import ErrorEnvelope
from src.api.schemas.oauth import (
    OAuthCompleteSignupRequest,
    OAuthExchangeRequest,
    OAuthSignupInfoRequest,
    OAuthSignupInfoResponse,
)
from src.api.security.jwt import JWTService
from src.api.security.oauth_tx_cookie import (
    OAUTH_TX_COOKIE,
    clear_oauth_tx_cookie,
    mint_oauth_tx_cookie,
)
from src.api.services.auth import EmailAlreadyExistsError
from src.api.services.legal.acceptance import RequestContext
from src.api.services.oauth.errors import (
    AccountInactiveError,
    FlowExpiredError,
    IdentityConflictError,
    InvalidHandoffError,
    InvalidSignupTicketError,
    OAuthCancelledError,
    OAuthError,
    OAuthFailedError,
    OAuthProviderNotEnabledError,
)
from src.api.services.oauth.pkce import generate_opaque_token
from src.api.services.oauth.service import CallbackRedirect, OAuthService
from src.core.config import Settings
from src.core.enums import OAuthErrorCode
from src.core.product import OAuthProvider, ProductConfig

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)

# The redirect responses carry one-time values (state in the provider URL,
# code/ticket in the fragment) — never cache them or leak them via Referer.
_REDIRECT_HEADERS: Final = {"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"}

TxBinding = Annotated[str | None, Parameter(cookie=OAUTH_TX_COOKIE, required=False)]


def _parse_provider(provider: str) -> OAuthProvider:
    try:
        return OAuthProvider(provider)
    except ValueError as exc:
        raise NotFoundException(detail="Unknown sign-in provider") from exc


def _error(code: str, message: str, status_code: int) -> Response[Any]:
    return Response(
        content=ErrorEnvelope(error=code, message=message, status_code=status_code),
        status_code=status_code,
    )


class OAuthController(Controller):
    """OAuth sign-in: authorize → provider → callback → exchange / signup."""

    path = "/v1/auth/oauth"
    tags: Sequence[str] | None = ("Authentication",)

    @get("/{provider:str}/authorize", status_code=HTTP_302_FOUND)
    async def authorize(
        self,
        provider: str,
        product_config: ProductConfig,
        oauth_service: OAuthService,
        settings: Settings,
        return_to: str | None = None,
    ) -> Redirect | Response[ErrorEnvelope]:
        """Start a sign-in: 302 to the provider, setting the ``apex_oauth_tx`` binding cookie.

        Navigate here (top-level, not fetch). 404 when the provider isn't
        enabled for this product; 400 for a ``return_to`` that isn't a
        same-origin path.
        """
        parsed = _parse_provider(provider)
        binding = generate_opaque_token()
        try:
            url = await oauth_service.start(
                product=product_config, provider=parsed, return_to=return_to, binding=binding
            )
        except OAuthProviderNotEnabledError as exc:
            raise NotFoundException(detail="Unknown sign-in provider") from exc
        except ValueError:
            return _error(
                "invalid_return_to", "return_to must be a same-origin path", HTTP_400_BAD_REQUEST
            )
        return Redirect(
            path=url,
            status_code=HTTP_302_FOUND,
            headers=_REDIRECT_HEADERS,
            cookies=[
                mint_oauth_tx_cookie(
                    binding,
                    max_age=settings.oauth_flow_ttl_seconds,
                    secure=settings.content_cookie_secure,
                )
            ],
        )

    @get("/{provider:str}/callback", status_code=HTTP_302_FOUND)
    async def callback(
        self,
        provider: str,
        product_config: ProductConfig,
        oauth_service: OAuthService,
        session: AsyncSession,
        settings: Settings,
        binding: TxBinding = None,
        code: str | None = None,
        oauth_state: Annotated[str | None, Parameter(query="state", required=False)] = None,
        error: str | None = None,
    ) -> Redirect:
        """Provider redirect target. Always 302s to the frontend ``/auth/callback#…``.

        The fragment is ``result=login&code=…``, ``result=signup&ticket=…``
        (each with optional ``return_to``), or ``result=error&error=<code>``.
        """
        parsed = _parse_provider(provider)
        try:
            redirect = await _callback_redirect(
                oauth_service,
                session,
                product=product_config,
                provider=parsed,
                code=code,
                state=oauth_state,
                error=error,
                binding=binding,
            )
            url = redirect.url
            binding_value = cast("str", binding)  # resolve_callback rejects a missing binding
            # SameSite restricts sending a cookie, not setting one on this cross-site redirect.
            cookies = [
                mint_oauth_tx_cookie(
                    binding_value,
                    max_age=redirect.binding_max_age,
                    secure=settings.content_cookie_secure,
                )
            ]
        except OAuthError as exc:
            # Leave the cookie untouched and let it lapse: a failed callback may belong to another tab's live flow.
            logger.info(
                "auth.oauth.callback_error",
                product_id=product_config.slug,
                provider=parsed.value,
                error=exc.code.value,
            )
            url = oauth_service.error_redirect(product=product_config, error=exc.code)
            cookies = []
        except Exception:
            # Never JSON from a browser navigation: roll back and report a
            # generic failure to the frontend.
            await session.rollback()
            logger.exception(
                "auth.oauth.callback_failed", product_id=product_config.slug, provider=parsed.value
            )
            url = oauth_service.error_redirect(
                product=product_config, error=OAuthErrorCode.OAUTH_FAILED
            )
            cookies = []
        return Redirect(
            path=url, status_code=HTTP_302_FOUND, headers=_REDIRECT_HEADERS, cookies=cookies
        )

    @post("/exchange", status_code=HTTP_200_OK)
    async def exchange(
        self,
        data: Annotated[OAuthExchangeRequest, Body()],
        oauth_service: OAuthService,
        jwt_service: JWTService,
        product_id: str,
        product_config: ProductConfig,
        settings: Settings,
        request_context: RequestContext,
        binding: TxBinding = None,
    ) -> Response[TokenResponse | ErrorEnvelope]:
        """Redeem the login handoff code: ``TokenResponse`` + content cookie; clears ``apex_oauth_tx``.

        Send with ``credentials: 'include'`` so the binding cookie is presented.
        """
        try:
            user_id, tokens = await oauth_service.exchange(
                product=product_config, code=data.code, binding=binding, context=request_context
            )
        except InvalidHandoffError:
            return _error(
                OAuthErrorCode.INVALID_HANDOFF.value,
                "This sign-in link is invalid or has expired. Please sign in again.",
                HTTP_400_BAD_REQUEST,
            )
        except AccountInactiveError:
            return _error(
                OAuthErrorCode.ACCOUNT_INACTIVE.value,
                "Account has been deactivated",
                HTTP_401_UNAUTHORIZED,
            )
        return build_token_response(
            user_id=user_id,
            product_id=product_id,
            tokens=tokens,
            jwt_service=jwt_service,
            settings=settings,
            product_config=product_config,
            status_code=HTTP_200_OK,
            extra_cookies=[clear_oauth_tx_cookie(secure=settings.content_cookie_secure)],
        )

    @post("/signup-info", status_code=HTTP_200_OK)
    async def signup_info(
        self,
        data: Annotated[OAuthSignupInfoRequest, Body()],
        oauth_service: OAuthService,
        product_config: ProductConfig,
        binding: TxBinding = None,
    ) -> Response[OAuthSignupInfoResponse | ErrorEnvelope]:
        """The pending signup's email and provider (non-consuming; safe to repeat)."""
        try:
            pending = await oauth_service.signup_info(
                product=product_config, ticket=data.ticket, binding=binding
            )
        except InvalidSignupTicketError:
            return _invalid_ticket()
        return Response(
            content=OAuthSignupInfoResponse(email=pending.email, provider=pending.provider.value),
            status_code=HTTP_200_OK,
        )

    @post("/complete-signup", status_code=HTTP_201_CREATED)
    async def complete_signup(
        self,
        data: Annotated[OAuthCompleteSignupRequest, Body()],
        oauth_service: OAuthService,
        jwt_service: JWTService,
        product_id: str,
        product_config: ProductConfig,
        settings: Settings,
        request_context: RequestContext,
        binding: TxBinding = None,
    ) -> Response[TokenResponse | ErrorEnvelope]:
        """Create the account after legal acceptance: 201 ``TokenResponse``; clears ``apex_oauth_tx``.

        ``accepted_documents`` errors map to 422 ``legal_acceptance_incomplete``
        / 409 ``legal_version_stale`` via the global handlers — before the
        ticket is consumed, so the client can retry with the current documents.
        """
        try:
            user, tokens = await oauth_service.complete_signup(
                product=product_config,
                ticket=data.ticket,
                binding=binding,
                accepted_documents=data.accepted_documents,
                display_name=data.display_name,
                context=request_context,
            )
        except InvalidSignupTicketError:
            return _invalid_ticket()
        except EmailAlreadyExistsError as exc:
            return _error("email_exists", str(exc), HTTP_400_BAD_REQUEST)
        except IdentityConflictError:
            return _error(
                OAuthErrorCode.IDENTITY_CONFLICT.value,
                "This sign-in is already linked to another account.",
                HTTP_409_CONFLICT,
            )
        return build_token_response(
            user_id=user.id,
            product_id=product_id,
            tokens=tokens,
            jwt_service=jwt_service,
            settings=settings,
            product_config=product_config,
            status_code=HTTP_201_CREATED,
            extra_cookies=[clear_oauth_tx_cookie(secure=settings.content_cookie_secure)],
        )


async def _callback_redirect(
    oauth_service: OAuthService,
    session: AsyncSession,
    *,
    product: ProductConfig,
    provider: OAuthProvider,
    code: str | None,
    state: str | None,
    error: str | None,
    binding: str | None,
) -> CallbackRedirect:
    """The success redirect for a callback, or raise the ``OAuthError`` to report."""
    if error is not None:
        raise OAuthCancelledError if error == "access_denied" else OAuthFailedError
    if not code or not state:
        raise FlowExpiredError
    outcome, return_to = await oauth_service.resolve_callback(
        product=product, provider=provider, state=state, code=code, binding=binding
    )
    if binding is None:  # unreachable: resolve_callback rejects a missing binding
        raise FlowExpiredError
    # Commit the link / last-login BEFORE handing out a code for it.
    await session.commit()
    return await oauth_service.issue_redirect(
        product=product, outcome=outcome, return_to=return_to, binding=binding
    )


def _invalid_ticket() -> Response[Any]:
    return _error(
        OAuthErrorCode.INVALID_SIGNUP_TICKET.value,
        "This sign-up session is invalid or has expired. Please start again.",
        HTTP_400_BAD_REQUEST,
    )
