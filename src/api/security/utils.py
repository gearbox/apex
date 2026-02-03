import hashlib
import secrets


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
