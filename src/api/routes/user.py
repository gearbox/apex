"""User profile API routes."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Annotated
from uuid import UUID

from litestar import Controller, Request, Response, delete, get, patch, post
from litestar.di import Provide
from litestar.exceptions import NotAuthorizedException, NotFoundException
from litestar.params import Body, Parameter
from litestar.status_codes import (
    HTTP_200_OK,
    HTTP_400_BAD_REQUEST,
)

from src.api.schemas.auth import AuthErrorResponse, MessageResponse
from src.api.schemas.user import (
    ChangePasswordRequest,
    DeleteAccountResponse,
    UpdateProfileRequest,
    UserJobsResponse,
    UserProfileResponse,
    UserStatsResponse,
)
from src.api.security import auth_guard
from src.api.services.auth import AuthService
from src.api.services.user import (
    EmailAlreadyExistsError,
    InvalidPasswordError,
    UserNotFoundError,
    UserService,
)

logger = logging.getLogger(__name__)


async def get_current_user_id(request: Request) -> UUID:
    """Extract current user ID from request state.

    Args:
        request: Litestar request.

    Returns:
        User ID.

    Raises:
        NotAuthorizedException: If not authenticated.
    """
    user_id = request.state.get("user_id")
    if user_id is None:
        raise NotAuthorizedException(detail="Not authenticated")
    return user_id


class UserController(Controller):
    """User profile management endpoints."""

    path = "/api/v1/users"
    tags: Sequence[str] | None = ["Users"]
    guards = [auth_guard]
    dependencies = {"current_user_id": Provide(get_current_user_id)}

    @get("/me")
    async def get_profile(
        self,
        current_user_id: UUID,
        user_service: UserService,
    ) -> Response[UserProfileResponse | AuthErrorResponse]:
        """Get current user's profile."""
        try:
            profile = await user_service.get_profile(current_user_id)
            return Response(content=profile, status_code=HTTP_200_OK)

        except UserNotFoundError as e:
            raise NotFoundException(detail="User not found") from e

    @patch("/me")
    async def update_profile(
        self,
        current_user_id: UUID,
        data: Annotated[UpdateProfileRequest, Body()],
        user_service: UserService,
    ) -> Response[UserProfileResponse | AuthErrorResponse]:
        """Update current user's profile."""
        try:
            profile = await user_service.update_profile(
                current_user_id,
                display_name=data.display_name,
                email=data.email,
            )
            return Response(content=profile, status_code=HTTP_200_OK)

        except UserNotFoundError as e:
            raise NotFoundException(detail="User not found") from e

        except EmailAlreadyExistsError as e:
            return Response(
                content=AuthErrorResponse(
                    error="email_exists",
                    error_description=str(e),
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @post("/me/password")
    async def change_password(
        self,
        current_user_id: UUID,
        data: Annotated[ChangePasswordRequest, Body()],
        user_service: UserService,
    ) -> Response[MessageResponse | AuthErrorResponse]:
        """Change current user's password.

        All existing sessions will be invalidated.
        """
        try:
            await user_service.change_password(
                current_user_id,
                current_password=data.current_password,
                new_password=data.new_password,
            )
            return Response(
                content=MessageResponse(
                    message="Password changed successfully. Please log in again."
                ),
                status_code=HTTP_200_OK,
            )

        except UserNotFoundError as e:
            raise NotFoundException(detail="User not found") from e

        except InvalidPasswordError:
            return Response(
                content=AuthErrorResponse(
                    error="invalid_password",
                    error_description="Current password is incorrect",
                ),
                status_code=HTTP_400_BAD_REQUEST,
            )

    @delete("/me", status_code=HTTP_200_OK)
    async def delete_account(
        self,
        current_user_id: UUID,
        user_service: UserService,
    ) -> Response[DeleteAccountResponse]:
        """Deactivate current user's account.

        This is a soft delete - the account can be recovered.
        All sessions will be invalidated.
        """
        try:
            deactivated_at = await user_service.deactivate_account(current_user_id)
            return Response(
                content=DeleteAccountResponse(
                    message="Account has been deactivated",
                    deactivated_at=deactivated_at,
                ),
                status_code=HTTP_200_OK,
            )

        except UserNotFoundError as e:
            raise NotFoundException(detail="User not found") from e

    @get("/me/stats")
    async def get_stats(
        self,
        current_user_id: UUID,
        user_service: UserService,
    ) -> Response[UserStatsResponse]:
        """Get current user's statistics."""
        try:
            stats = await user_service.get_stats(current_user_id)
            return Response(content=stats, status_code=HTTP_200_OK)

        except UserNotFoundError as e:
            raise NotFoundException(detail="User not found") from e

    @get("/me/jobs")
    async def get_jobs(
        self,
        current_user_id: UUID,
        user_service: UserService,
        limit: Annotated[int, Parameter(ge=1, le=100)] = 50,
        offset: Annotated[int, Parameter(ge=0)] = 0,
    ) -> Response[UserJobsResponse]:
        """Get current user's generation jobs."""
        try:
            jobs = await user_service.get_jobs(
                current_user_id,
                limit=limit,
                offset=offset,
            )
            return Response(content=jobs, status_code=HTTP_200_OK)

        except UserNotFoundError as e:
            raise NotFoundException(detail="User not found") from e

    @post("/me/logout-all")
    async def logout_all(
        self,
        current_user_id: UUID,
        auth_service: AuthService,
    ) -> Response[MessageResponse]:
        """Logout from all devices.

        Invalidates all refresh tokens for the current user.
        """
        count = await auth_service.logout_all(current_user_id)
        return Response(
            content=MessageResponse(message=f"Logged out from {count} session(s)"),
            status_code=HTTP_200_OK,
        )
