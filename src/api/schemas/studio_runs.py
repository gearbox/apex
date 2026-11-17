"""Request / response DTOs for studio planner runs (``docs/contracts/studio-contract.md`` §11).

The request Structs live in ``src/studio/run_requests.py`` (the planner reads them and may not
import this layer) and are re-exported here for the routes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

import msgspec

from src.core.enums import (
    StudioRunKind,
    StudioRunStage,
    StudioRunStatus,
)
from src.studio.run_requests import (
    LyricsRunRequest,
    PlanRunRequest,
    ReviseLyricsRunRequest,
    RunRequest,
)

__all__ = [
    "LyricsRunRequest",
    "PlanRunRequest",
    "ReviseLyricsRunRequest",
    "RunOut",
    "RunRequest",
    "RunWarningOut",
]


class RunWarningOut(msgspec.Struct, kw_only=True):
    """A non-fatal note about a succeeded run: a stable ``code`` and, where it applies, a ``path``
    and a ``count``. Codes, paths and numbers only - never text."""

    code: str
    path: str | None = None
    count: int | None = None


class RunOut(msgspec.Struct, kw_only=True):
    """A run. Never the request text, the context snapshot, the prompts or token counts."""

    id: UUID
    track_id: UUID
    kind: StudioRunKind
    status: StudioRunStatus
    stage: StudioRunStage | None
    revision: int
    """Strictly increases with every status or stage change (see ``studio.run.updated``)."""
    price_tokens: int
    model_key: str
    result_plan_version_id: UUID | None
    result_lyrics_version_id: UUID | None
    warnings: list[RunWarningOut]
    error_code: str | None
    error_detail: dict[str, Any] | None
    """Issue paths and codes only (``{"issues": [{"path", "code"}]}``) - never model or user text."""
    llm_calls: int
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
