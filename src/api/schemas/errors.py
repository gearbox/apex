"""Unified error response envelope."""

from __future__ import annotations

from typing import Any

import msgspec


class ErrorEnvelope(msgspec.Struct, kw_only=True):
    """Unified error shape for all non-2xx responses.

    Every error response from the API has this exact shape so frontend code
    can use a single typed error parser.
    """

    error: str  # machine-readable code, e.g. "not_found", "insufficient_balance"
    message: str  # human-readable, safe to display in UI
    status_code: int  # mirrors the HTTP status (useful in batch/aggregate responses)
    detail: dict[str, Any] | None = None  # optional structured context
