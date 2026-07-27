"""Authentication API routes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any
from uuid import UUID

import msgspec
import structlog
from litestar import Controller, Request, Response, get, post
from litestar.di import Provide
from litestar.params import Body
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
)
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.dependencies.auth import get_current_user_id
from src.api.schemas.auth import (
    ContentCookieResponse,
    ForgotPasswordRequest,
    LoginRequest,
    MessageResponse,
    RefreshTokenRequest,
    RegisterRequest,
    ResetPasswordRequest,
    TokenResponse,
    VerifyEmailRequest,
)
from src.api.schemas.errors import ErrorEnvelope
from src.api.security import CLEAR_SITE_DATA_HEADER, auth_guard, extract_token_from_header
from src.api.security.content_cookie import (
    clear_content_cookie,
    effective_cookie_domain,
    mint_content_cookie,
)
from src.api.security.jwt import JWTService
from src.api.services.auth import (
    AuthService,
    EmailAlreadyExistsError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    TokenReuseDetectedError,
    UserInactiveError,
)
from src.api.services.email_verification import (
    EmailVerificationService,
    InvalidTokenError,
    UserNotFoundError,
)
from src.api.services.token_revocation import TokenRevocationService
from src.core.config import Settings
from src.core.product import ProductConfig
from src.db.repositories.user import UserRepository

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = structlog.get_logger(__name__)


class ProductInfoResponse(msgspec.Struct, kw_only=True):
    """Product context returned by GET /v1/auth/product-info."""

    product: str
    """Product slug, e.g. 'vex' or 'synthara'."""

    display_name: str
    """Human-readable product name."""

    age_gate: str
    """Age verification requirement: 'none' | 'checkbox' | 'date_of_birth'."""

    allowed_auth_methods: list[str]
    """Authentication methods enabled for this product."""

    content_rating: str
    """Content rating: 'sfw' | 'permissive'."""

    payment_providers: list[str]
    """Enabled payment providers for this product."""


class AuthController(Controller):
    """Authentication endpoints."""

    path = "/v1/auth"
    tags: Sequence[str] | None = ("Authentication",)

    @post("/register", status_code=HTTP_201_CREATED)
    async def register(
        self,
        data: Annotated[RegisterRequest, Body()],
        auth_service: AuthService,
        jwt_service: JWTService,
        product_id: str,
        product_config: ProductConfig,
        settings: Settings,
    ) -> Response[TokenResponse | ErrorEnvelope]:
        """Register a new user account.

        Creates a new user and returns authentication tokens.
        """
        try:
            user, tokens = await auth_service.register(
                email=data.email,
                password=data.password,
                product_id=product_id,
                display_name=data.display_name,
            )

            content_cookie, content_cookie_expires_at = mint_content_cookie(
                user_id=user.id,
                product_id=product_id,
                jwt_service=jwt_service,
                settings=settings,
                product_config=product_config,
            )
            response: Response[TokenResponse | ErrorEnvelope] = Response(
                content=TokenResponse(
                    access_token=tokens.access_token,
                    refresh_token=tokens.refresh_token,
                    expires_in=tokens.expires_in,
                    expires_at=tokens.expires_at,
                    content_cookie_expires_at=content_cookie_expires_at,
                ),
                status_code=HTTP_201_CREATED,
                cookies=[content_cookie],
            )

        except EmailAlreadyExistsError as e:
            return Response(
                content=ErrorEnvelope(
                    error="email_exists",
                    message=str(e),
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        else:
            return response

    @post("/verify-email")
    async def verify_email(
        self,
        data: Annotated[VerifyEmailRequest, Body()],
        session: AsyncSession,
        email_verification_service: EmailVerificationService,
    ) -> Response[MessageResponse | ErrorEnvelope]:
        """Verify a user's email address using a token from the verification link.

        The token is single-use and expires after 24 hours.
        Returns 400 for invalid, expired, or already-used tokens.
        """
        try:
            await email_verification_service.verify_email(data.token, session=session)
            await session.commit()
            return Response(
                content=MessageResponse(message="Email verified successfully"),
                status_code=HTTP_200_OK,
            )
        except InvalidTokenError:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_token",
                    message="This verification link is invalid or has expired.",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post(
        "/resend-verification",
        guards=[auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    async def resend_verification(
        self,
        current_user_id: UUID,
        session: AsyncSession,
        email_verification_service: EmailVerificationService,
    ) -> Response[MessageResponse | ErrorEnvelope]:
        """Resend the email verification link to the authenticated user.

        Rate limiting should be applied externally (e.g. nginx / middleware).
        Returns 200 even if the email is already verified — silent no-op.
        """
        user_id: UUID = current_user_id

        user_repo = UserRepository(session)
        user = await user_repo.get_active_user(user_id)

        if user is None:
            return Response(
                content=ErrorEnvelope(
                    error="user_not_found",
                    message="User not found.",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

        if user.email_verified_at is not None:
            # Already verified — silent success, no new email
            return Response(
                content=MessageResponse(message="Email is already verified"),
                status_code=HTTP_200_OK,
            )

        try:
            await email_verification_service.send_verification_email(user_id, session=session)
            await session.commit()
            return Response(
                content=MessageResponse(message="Verification email sent"),
                status_code=HTTP_200_OK,
            )
        except UserNotFoundError:
            return Response(
                content=ErrorEnvelope(
                    error="user_not_found",
                    message="User not found.",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post(
        "/content-cookie",
        status_code=HTTP_200_OK,
        guards=[auth_guard],
        dependencies={"current_user_id": Provide(get_current_user_id)},
    )
    async def remint_content_cookie(
        self,
        current_user_id: UUID,
        jwt_service: JWTService,
        product_id: str,
        product_config: ProductConfig,
        settings: Settings,
    ) -> Response[ContentCookieResponse]:
        """Re-mint the content authentication cookie (apex_content).

        Bearer-only — requires a valid, unexpired access token. With the
        content cookie's TTL now measured in days (see
        Settings.content_cookie_ttl_hours), this endpoint's role has shifted:
        it no longer exists mainly to dodge a refresh-token rotation, but is
        the *proactive* keep-alive path — clients schedule a re-mint from the
        `expires_at` this returns (or from TokenResponse.content_cookie_expires_at)
        well before the cookie actually lapses, rather than waiting for a 401.
        """
        content_cookie, expires_at = mint_content_cookie(
            user_id=current_user_id,
            product_id=product_id,
            jwt_service=jwt_service,
            settings=settings,
            product_config=product_config,
        )
        return Response(
            content=ContentCookieResponse(expires_at=expires_at),
            status_code=HTTP_200_OK,
            cookies=[content_cookie],
        )

    @post("/login")
    async def login(
        self,
        request: Request[Any, Any, Any],
        data: Annotated[LoginRequest, Body()],
        auth_service: AuthService,
        jwt_service: JWTService,
        product_id: str,
        product_config: ProductConfig,
        settings: Settings,
    ) -> Response[TokenResponse | ErrorEnvelope]:
        """Authenticate user and return tokens.

        Returns access and refresh tokens for valid credentials.
        """
        try:
            user_agent = request.headers.get("user-agent")
            # Get client IP (considering proxy headers)
            ip_address = (
                request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or request.client.host
                if request.client
                else None
            )

            user, tokens = await auth_service.login(
                email=data.email,
                password=data.password,
                product_id=product_id,
                user_agent=user_agent,
                ip_address=ip_address,
            )

            content_cookie, content_cookie_expires_at = mint_content_cookie(
                user_id=user.id,
                product_id=product_id,
                jwt_service=jwt_service,
                settings=settings,
                product_config=product_config,
            )
            response: Response[TokenResponse | ErrorEnvelope] = Response(
                content=TokenResponse(
                    access_token=tokens.access_token,
                    refresh_token=tokens.refresh_token,
                    expires_in=tokens.expires_in,
                    expires_at=tokens.expires_at,
                    content_cookie_expires_at=content_cookie_expires_at,
                ),
                status_code=HTTP_200_OK,
                cookies=[content_cookie],
            )

        except InvalidCredentialsError:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_credentials",
                    message="Invalid email or password",
                    status_code=HTTP_401_UNAUTHORIZED,
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        except UserInactiveError:
            return Response(
                content=ErrorEnvelope(
                    error="account_inactive",
                    message="Account has been deactivated",
                    status_code=HTTP_401_UNAUTHORIZED,
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        else:
            return response

    @post("/refresh")
    async def refresh_tokens(
        self,
        request: Request[Any, Any, Any],
        data: Annotated[RefreshTokenRequest, Body()],
        auth_service: AuthService,
        jwt_service: JWTService,
        product_id: str,
        product_config: ProductConfig,
        settings: Settings,
    ) -> Response[TokenResponse | ErrorEnvelope]:
        """Refresh access token using refresh token.

        Implements token rotation - old refresh token is invalidated.
        """
        try:
            user_agent = request.headers.get("user-agent")
            ip_address = (
                request.headers.get("x-forwarded-for", "").split(",")[0].strip()
                or request.client.host
                if request.client
                else None
            )

            tokens, user_id = await auth_service.refresh_tokens(
                data.refresh_token,
                user_agent=user_agent,
                ip_address=ip_address,
            )

            content_cookie, content_cookie_expires_at = mint_content_cookie(
                user_id=user_id,
                product_id=product_id,
                jwt_service=jwt_service,
                settings=settings,
                product_config=product_config,
            )
            response: Response[TokenResponse | ErrorEnvelope] = Response(
                content=TokenResponse(
                    access_token=tokens.access_token,
                    refresh_token=tokens.refresh_token,
                    expires_in=tokens.expires_in,
                    expires_at=tokens.expires_at,
                    content_cookie_expires_at=content_cookie_expires_at,
                ),
                status_code=HTTP_200_OK,
                cookies=[content_cookie],
            )

        except InvalidRefreshTokenError:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_token",
                    message="Refresh token is invalid or expired",
                    status_code=HTTP_401_UNAUTHORIZED,
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        except TokenReuseDetectedError as e:
            return Response(
                content=ErrorEnvelope(
                    error="token_reuse_detected",
                    message=str(e),
                    status_code=HTTP_401_UNAUTHORIZED,
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        except UserInactiveError:
            return Response(
                content=ErrorEnvelope(
                    error="account_inactive",
                    message="Account has been deactivated",
                    status_code=HTTP_401_UNAUTHORIZED,
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        else:
            return response

    @post("/logout")
    async def logout(
        self,
        request: Request[Any, Any, Any],
        data: Annotated[RefreshTokenRequest, Body()],
        auth_service: AuthService,
        jwt_service: JWTService,
        token_revocation_service: TokenRevocationService,
        product_config: ProductConfig,
        settings: Settings,
    ) -> Response[MessageResponse]:
        """Logout by invalidating the refresh token and denylisting the
        presenting access token's own jti (issue #142).

        This route is intentionally unguarded — the Authorization header is
        optional here (the access token may already be expired), so it's
        decoded manually rather than via auth_guard. When present and valid,
        its jti is denylisted for its remaining lifetime; when absent,
        expired, or otherwise undecodable, only the refresh token is revoked
        (identical to pre-#142 behavior).

        Known limitation: this device's `apex_content` cookie is cleared
        client-side (via the Set-Cookie below) but is **not** revoked
        server-side — the cookie is scoped `Path=/v1/content`, so it's never
        sent to this endpoint, and it carries a different `jti` than the
        access token being denylisted above. For an honest logout this is
        harmless (the cookie is gone from the browser); for a *stolen*
        cookie it means the credential keeps working until it expires (up to
        `CONTENT_COOKIE_TTL_HOURS`) or until a bulk revocation occurs
        (logout-all, password change/reset, deactivation). Do not "fix" this
        by calling `revoke_user_sessions` here — that would log the user out
        of every other device, which single-device logout must not do. Users
        who suspect theft should use logout-all or change/reset their
        password.

        The response also sends `Clear-Site-Data: "cache", "storage"`,
        instructing the browser to drop its HTTP disk cache (and storage)
        for this API origin — the origin content URLs (`/v1/content/...`)
        are served from. This closes the browser-cache residue left behind
        after logout (cached content responses were otherwise readable
        until natural cache eviction), rather than relying on a
        client-side workaround.

        Deliberate contract: this endpoint always responds 200 with the
        Clear-Site-Data header, regardless of whether `refresh_token` was
        valid, unknown, or already revoked — `auth_service.logout`'s return
        value is intentionally ignored below. A client that discovers its
        session was revoked remotely (logout-all/password change/reset
        elsewhere) has no way to clear this origin's HTTP cache itself, so
        it fires a best-effort logout purely to receive this header. Making
        the response depend on token validity would both break that
        recovery path and leak whether a given refresh token was ever
        valid — the uniform 200 is also the correct privacy posture. Do not
        "fix" this by 401ing on an unknown/invalid token.
        """
        await auth_service.logout(data.refresh_token)

        if raw_bearer := extract_token_from_header(request.headers.get("authorization")):
            payload = jwt_service.decode_access_token(raw_bearer)
            if payload is not None and not await token_revocation_service.revoke_token(
                payload.jti, payload.exp
            ):
                # Redis is configured but the write failed — a genuine
                # degradation, not the documented Redis-unset no-op (F5).
                # Logout still succeeds: the refresh token is already
                # revoked above, which is the primary guarantee.
                logger.error("auth.jti_denylist_failed", jti=payload.jti)

        return Response(
            content=MessageResponse(message="Successfully logged out"),
            status_code=HTTP_200_OK,
            cookies=[
                clear_content_cookie(
                    domain=effective_cookie_domain(settings, product_config),
                    secure=settings.content_cookie_secure,
                )
            ],
            headers=CLEAR_SITE_DATA_HEADER,
        )

    @post("/forgot-password")
    async def forgot_password(
        self,
        request: Request[Any, Any, Any],
        data: Annotated[ForgotPasswordRequest, Body()],
        session: AsyncSession,
        email_verification_service: EmailVerificationService,
    ) -> Response[MessageResponse]:
        """Request a password reset email.

        **Always returns 200** — does not reveal whether the email is registered.
        The reset link expires in 30 minutes.
        """
        ip_address = request.headers.get("x-forwarded-for", "").split(",")[0].strip() or (
            request.client.host if request.client else None
        )

        # Fire and forget — any error is logged inside the service
        try:
            await email_verification_service.send_password_reset_email(
                data.email,
                session=session,
                ip_address=ip_address,
            )
            await session.commit()
        except Exception:
            logger.exception("auth.forgot_password_failed")

        return Response(
            content=MessageResponse(
                message="If that email is registered, you'll receive a reset link shortly."
            ),
            status_code=HTTP_200_OK,
        )

    @post("/reset-password")
    async def reset_password(
        self,
        data: Annotated[ResetPasswordRequest, Body()],
        session: AsyncSession,
        email_verification_service: EmailVerificationService,
    ) -> Response[MessageResponse | ErrorEnvelope]:
        """Consume a password reset token and update the password.

        Revokes all active refresh tokens (forces re-login on all devices).
        Returns 400 for invalid or expired tokens. This is the
        compromised-account recovery path, so on success the response also
        sends `Clear-Site-Data: "cache", "storage"` — the caller's device
        ends its own session here too, and the account is being recovered
        because it's suspected compromised.
        """
        try:
            await email_verification_service.reset_password(
                data.token,
                data.new_password,
                session=session,
            )
            await session.commit()
            return Response(
                content=MessageResponse(
                    message="Password updated successfully. Please log in with your new password."
                ),
                status_code=HTTP_200_OK,
                headers=CLEAR_SITE_DATA_HEADER,
            )
        except InvalidTokenError:
            return Response(
                content=ErrorEnvelope(
                    error="invalid_token",
                    message="This reset link is invalid or has expired.",
                    status_code=HTTP_400_BAD_REQUEST,
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @get("/product-info")
    async def product_info(
        self,
        product_config: ProductConfig,
    ) -> ProductInfoResponse:
        """Return product context for the current origin.

        Public endpoint — no authentication required.
        The frontend calls this on load to configure the UI
        (age gate requirements, allowed auth methods, content rating).
        """
        return ProductInfoResponse(
            product=product_config.slug,
            display_name=product_config.display_name,
            age_gate=product_config.age_gate.value,
            allowed_auth_methods=[m.value for m in product_config.allowed_auth_methods],
            content_rating=product_config.content_policy.rating.value,
            payment_providers=[p.value for p in product_config.payment_providers],
        )
