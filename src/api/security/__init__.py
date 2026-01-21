"""Security module for authentication and authorization."""

from .guards import (
    AuthenticatedUser,
    auth_guard,
    extract_token_from_header,
    optional_auth_guard,
)
from .jwt import (
    InvalidTokenError,
    JWTConfig,
    JWTService,
    TokenExpiredError,
    TokenPayload,
)
from .password import (
    PasswordService,
    generate_token,
    get_password_service,
    hash_token,
)

__all__ = [
    # Guards
    "AuthenticatedUser",
    "auth_guard",
    "extract_token_from_header",
    "optional_auth_guard",
    # JWT
    "InvalidTokenError",
    "JWTConfig",
    "JWTService",
    "TokenExpiredError",
    "TokenPayload",
    # Password
    "PasswordService",
    "generate_token",
    "get_password_service",
    "hash_token",
]
