"""Authentication API routes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated

from litestar import Controller, Request, Response, post
from litestar.params import Body
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_201_CREATED,
    HTTP_400_BAD_REQUEST,
    HTTP_401_UNAUTHORIZED,
)

from src.api.schemas.auth import (
    AuthErrorResponse,
    LoginRequest,
    MessageResponse,
    RefreshTokenRequest,
    RegisterRequest,
    TokenResponse,
)
from src.api.services.auth import (
    AuthService,
    EmailAlreadyExistsError,
    InvalidCredentialsError,
    InvalidRefreshTokenError,
    TokenReuseDetectedError,
    UserInactiveError,
)

logger = logging.getLogger(__name__)


class AuthController(Controller):
    """Authentication endpoints."""

    path = "/api/v1/auth"
    tags: Sequence[str] | None = ["Authentication"]

    @post("/register", status_code=HTTP_201_CREATED)
    async def register(
        self,
        data: Annotated[RegisterRequest, Body()],
        auth_service: AuthService,
    ) -> Response[TokenResponse | AuthErrorResponse]:
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
                content=AuthErrorResponse(
                    error="email_exists",
                    error_description=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/login")
    async def login(
        self,
        request: Request,
        data: Annotated[LoginRequest, Body()],
        auth_service: AuthService,
    ) -> Response[TokenResponse | AuthErrorResponse]:
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
                content=AuthErrorResponse(
                    error="invalid_credentials",
                    error_description="Invalid email or password",
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        except UserInactiveError:
            return Response(
                content=AuthErrorResponse(
                    error="account_inactive",
                    error_description="Account has been deactivated",
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

    @post("/refresh")
    async def refresh_tokens(
        self,
        request: Request,
        data: Annotated[RefreshTokenRequest, Body()],
        auth_service: AuthService,
    ) -> Response[TokenResponse | AuthErrorResponse]:
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
                content=AuthErrorResponse(
                    error="invalid_token",
                    error_description="Refresh token is invalid or expired",
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        except TokenReuseDetectedError as e:
            return Response(
                content=AuthErrorResponse(
                    error="token_reuse_detected",
                    error_description=str(e),
                ),
                status_code=HTTP_401_UNAUTHORIZED,
            )

        except UserInactiveError:
            return Response(
                content=AuthErrorResponse(
                    error="account_inactive",
                    error_description="Account has been deactivated",
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
