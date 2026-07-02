"""Password hashing utilities using argon2."""

from __future__ import annotations

import anyio
import anyio.to_thread
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerifyMismatchError

# Argon2 hashing (64 MiB, time_cost=3) costs ~100-300ms of CPU per call. Run
# off the event loop via anyio.to_thread, bounded by this process-wide
# capacity limiter so a login burst can't exhaust the default thread pool
# (default limiter capacity is 40, shared with every other to_thread caller).
_HASH_CONCURRENCY = anyio.CapacityLimiter(4)


class PasswordService:
    """Password hashing and verification using Argon2id.

    Argon2id is the recommended algorithm for password hashing,
    combining resistance to both side-channel and GPU attacks.

    ``hash``/``verify`` are synchronous — safe for CLI/tests/offline use, but
    each call blocks the event loop for ~100-300ms. Request-handling code
    (AuthService et al.) MUST use ``ahash``/``averify`` instead, which offload
    the same work to a worker thread.

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
        """Hash a password (blocking — see class docstring).

        Args:
            password: Plain text password.

        Returns:
            Argon2 hash string.
        """
        return self._hasher.hash(password)

    async def ahash(self, password: str) -> str:
        """Hash a password off the event loop.

        Args:
            password: Plain text password.

        Returns:
            Argon2 hash string.
        """
        return await anyio.to_thread.run_sync(
            self._hasher.hash, password, limiter=_HASH_CONCURRENCY
        )

    def verify(self, hash: str, password: str) -> bool:
        """Verify a password against a hash (blocking — see class docstring).

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

    async def averify(self, hash: str, password: str) -> bool:
        """Verify a password against a hash off the event loop.

        Args:
            hash: Argon2 hash string.
            password: Plain text password to verify.

        Returns:
            True if password matches, False otherwise.
        """
        return await anyio.to_thread.run_sync(
            self._verify_sync, hash, password, limiter=_HASH_CONCURRENCY
        )

    def _verify_sync(self, hash: str, password: str) -> bool:
        """Thread-target for ``averify`` — reuses the sync verify() logic."""
        return self.verify(hash, password)

    def needs_rehash(self, hash: str) -> bool:
        """Check if a hash needs to be rehashed.

        This is useful when upgrading hashing parameters.

        Args:
            hash: Existing hash to check.

        Returns:
            True if hash should be rehashed with current parameters.
        """
        return self._hasher.check_needs_rehash(hash)
