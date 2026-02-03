"""Password hashing utilities using argon2."""

from __future__ import annotations

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError


class PasswordService:
    """Password hashing and verification using Argon2id.

    Argon2id is the recommended algorithm for password hashing,
    combining resistance to both side-channel and GPU attacks.

    Example:
        >>> service = PasswordService()
        >>> hashed = service.hash("secret123")
        >>> service.verify(hashed, "secret123")
        True
        >>> service.verify(hashed, "wrong")
        False
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
