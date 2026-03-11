"""API schemas for generation model enable/disable feature."""

from __future__ import annotations

from datetime import datetime

import msgspec

__all__ = ["GenerationModelResponse", "ModelListResponse", "SetModelEnabledRequest"]


class GenerationModelResponse(msgspec.Struct, kw_only=True):
    """Response schema for a single generation model."""

    model_key: str
    provider: str
    name: str
    description: str
    is_enabled: bool
    created_at: datetime
    updated_at: datetime


class SetModelEnabledRequest(msgspec.Struct, forbid_unknown_fields=True, kw_only=True):
    """Request body for toggling a model's is_enabled flag."""

    is_enabled: bool


class ModelListResponse(msgspec.Struct, kw_only=True):
    """Response schema for a list of generation models."""

    items: list[GenerationModelResponse]
    total: int
