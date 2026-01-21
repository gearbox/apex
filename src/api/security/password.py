"""Password hashing utilities using argon2."""

from __future__ import annotations

import hashlib
import secrets

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError


class PasswordService:
    """Password hashing and verification using Argon2id.

    Argon2id is the recommended algorithm for password hashing,
    combining resistance to both side-channel and GPU attacks.
    """

    def __init__(
        self,
        time_cost: int = 3,
        memory_cost: int = 65536,  # 64 MiB
        parallelism: int = 4,
    ) -> None:
        """Initialize password hasher.

        Args:
            time_cost: Number of iterations.
            memory_cost: Memory usage in KiB.
            parallelism: Number of parallel threads.
        """
        self._hasher = PasswordHasher(
            time_cost=time_cost,
            memory_cost=memory_cost,
            parallelism=parallelism,
        )

    def hash(self, password: str) -> str:
        """Hash a password.

        Args:
            password: Plain text password.

        Returns:
            Argon2 hash string.
        """
        return self._hasher.hash(password)

    def verify(self, hash: str, password: str) -> bool:
        """Verify a password against a hash.

        Args:
            hash: Argon2 hash string.
            password: Plain text password to verify.

        Returns:
            True if password matches, False otherwise.
        """
        try:
            self._hasher.verify(hash, password)
            return True
        except (VerifyMismatchError, InvalidHashError):
            return False

    def needs_rehash(self, hash: str) -> bool:
        """Check if a hash needs to be rehashed.

        This is useful when upgrading hashing parameters.

        Args:
            hash: Existing hash to check.

        Returns:
            True if hash should be rehashed with current parameters.
        """
        return self._hasher.check_needs_rehash(hash)


def generate_token(nbytes: int = 32) -> str:
    """Generate a cryptographically secure random token.

    Args:
        nbytes: Number of random bytes (default 32 = 256 bits).

    Returns:
        URL-safe base64 encoded token.
    """
    return secrets.token_urlsafe(nbytes)


def hash_token(token: str) -> str:
    """Hash a token for storage.

    Uses SHA-256 for fast, non-reversible hashing of tokens.
    Unlike passwords, tokens are already high-entropy random strings.

    Args:
        token: Token to hash.

    Returns:
        Hex-encoded SHA-256 hash.
    """
    return hashlib.sha256(token.encode()).hexdigest()


# Default instance
_password_service: PasswordService | None = None


def get_password_service() -> PasswordService:
    """Get the default password service instance.

    Returns:
        Singleton PasswordService.
    """
    global _password_service
    if _password_service is None:
        _password_service = PasswordService()
    return _password_service
