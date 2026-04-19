"""Service-layer DTOs for GPU session operations.

These are internal to the service layer. The HTTP-facing schemas (request/response
bodies) live in src/api/schemas/gpu_session.py and are defined in Phase 1F.
"""

from __future__ import annotations

from uuid import UUID

import msgspec


class StopConfirmation(msgspec.Struct, kw_only=True):
    """Returned by stop_session(confirmed=False).

    The UI should show this to the user and require explicit confirmation
    before calling stop_session(confirmed=True).
    """

    session_id: UUID
    model_type: str
    bundle_name: str
    vastai_gpu_name: str | None
    vastai_cost_per_hour_micros: int | None
    active_duration_seconds: int
    message: str
