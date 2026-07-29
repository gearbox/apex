"""Security module - authentication and authorization.

This module provides implementations only. For DI providers,
use src.api.dependencies.
"""

from .content_cookie import build_content_cookie, clear_content_cookie, effective_cookie_domain
from .guards import (
    AuthenticatedUser,
    auth_guard,
    content_auth_guard,
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
from .response_headers import CLEAR_SITE_DATA_HEADER
from .revocation_recheck import recheck_revocation_or_raise
from .utils import generate_token, hash_token

__all__ = [
    "CLEAR_SITE_DATA_HEADER",
    "AuthenticatedUser",
    "InvalidTokenError",
    "JWTConfig",
    "JWTService",
    "PasswordService",
    "TokenExpiredError",
    "TokenPayload",
    "auth_guard",
    "build_content_cookie",
    "clear_content_cookie",
    "content_auth_guard",
    "effective_cookie_domain",
    "extract_token_from_header",
    "generate_token",
    "hash_token",
    "optional_auth_guard",
    "recheck_revocation_or_raise",
]
