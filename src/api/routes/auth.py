"""Authentication API routes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

import structlog
from litestar import Controller, Request, Response, post
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
from src.api.security import auth_guard
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
from src.db.repositories.user import UserRepository

logger = structlog.get_logger(__name__)


class AuthController(Controller):
    """Authentication endpoints."""

    path = "/v1/auth"
    tags: Sequence[str] | None = ["Authentication"]

    @post("/register", status_code=HTTP_201_CREATED)
    async def register(
        self,
        data: Annotated[RegisterRequest, Body()],
        auth_service: AuthService,
    ) -> Response[TokenResponse | ErrorEnvelope]:
        """Register a new user account.

        Creates a new user and returns authentication tokens.
        """
        try:
            user, tokens = await auth_service.register(
                email=data.email,
                password=data.password,
                display_name=data.display_name,
            )

            return Response(
                content=TokenResponse(
                    access_token=tokens.access_token,
                    refresh_token=tokens.refresh_token,
                    expires_in=tokens.expires_in,
                    expires_at=tokens.expires_at,
                ),
                status_code=HTTP_201_CREATED,
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

    @post("/login")
    async def login(
        self,
        request: Request,
        data: Annotated[LoginRequest, Body()],
        auth_service: AuthService,
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
                user_agent=user_agent,
                ip_address=ip_address,
            )

            return Response(
                content=TokenResponse(
                    access_token=tokens.access_token,
                    refresh_token=tokens.refresh_token,
                    expires_in=tokens.expires_in,
                    expires_at=tokens.expires_at,
                ),
                status_code=HTTP_200_OK,
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

    @post("/refresh")
    async def refresh_tokens(
        self,
        request: Request,
        data: Annotated[RefreshTokenRequest, Body()],
        auth_service: AuthService,
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

            tokens = await auth_service.refresh_tokens(
                data.refresh_token,
                user_agent=user_agent,
                ip_address=ip_address,
            )

            return Response(
                content=TokenResponse(
                    access_token=tokens.access_token,
                    refresh_token=tokens.refresh_token,
                    expires_in=tokens.expires_in,
                    expires_at=tokens.expires_at,
                ),
                status_code=HTTP_200_OK,
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

    @post("/logout")
    async def logout(
        self,
        data: Annotated[RefreshTokenRequest, Body()],
        auth_service: AuthService,
    ) -> Response[MessageResponse]:
        """Logout by invalidating refresh token.

        The access token will remain valid until expiration.
        """
        await auth_service.logout(data.refresh_token)

        return Response(
            content=MessageResponse(message="Successfully logged out"),
            status_code=HTTP_200_OK,
        )

    @post("/forgot-password")
    async def forgot_password(
        self,
        request: Request,
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
        Returns 400 for invalid or expired tokens.
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
