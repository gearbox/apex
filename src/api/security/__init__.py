"""Security module - authentication and authorization.

This module provides implementations only. For DI providers,
use src.api.dependencies.
"""

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
from .password import PasswordService
from .utils import generate_token, hash_token

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
    "hash_token",
]
