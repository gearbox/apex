"""Authentication helpers for per-session node callback tokens."""

from __future__ import annotations

import hashlib
import hmac


def validate_callback_token(presented: str, stored_hash: str | None) -> bool:
    """Compare a node's callback token with its stored SHA-256 digest in constant time."""
    if not stored_hash:
        return False
    presented_hash = hashlib.sha256(presented.encode()).hexdigest()
    return hmac.compare_digest(presented_hash, stored_hash)
